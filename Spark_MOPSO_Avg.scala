package com.hadwinling

import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
 * @Classname Main
 * @Description TODO
 * @Date 2022/11/3 14:50
 * @Created by ${hadwinling}
 */
object Main extends Logging {
  var iter = 1 // 循环计数器
  val Wmax = 0.9
  val Wmin = 0.4
  var Vmax = 1.0
  var Vmin = -1.0
  var repository = 15 // 非支配解种群数量,存档大小
  //  var iter_max = 50 // 迭代次数
  val numberOfObjective = 2 // 目标数量
  var numberOfParticles = 50 //粒子数量
  var numberOfSubParticles = 50 //粒子数量
  var numberOfFeatures = 19 // numberOfFeatures 即有多少个决策变量（对应数据集中除了代表簇的那一列外还剩有多少列）
  var numberOfClusters = 7 // numClusters 表示一共聚成了多少个簇（即K 值）
  val numberOfKMeansIter = 30 // 设置 kmeans 的运行次数
  var LIndex = 10 // 最近邻的个数设置
  val numberOfMerge = 1 // 相隔 numberOfMerge 进行合并
  val shold = 0.0000000001 //阈值，用于判断聚类中心偏移量
  val mu = 0.1 // Mutation Rate


  /*
  * TODO
  *  数据并行，随机分区
  * */
  def main(args: Array[String]): Unit = {
    val dataName = args(0)
    val numberOfWorker = args(1).toInt
    val numberOfSubPop = args(2).toInt
    val iter_max = args(3).toInt
    val labelStartWithZero = args(4).toInt
    val inputHDFS: String = "hdfs://211.69.243.34:9000/linghuidong/input/" + dataName
    val outputHDFS: String = "hdfs://211.69.243.34:9000/linghuidong/output/" + dataName + "/randomPartition"

    Logger.getLogger("org").setLevel(Level.ERROR)

    val start = System.nanoTime
    // Create a SparkContext using every core of the cluster
    //        val conf = new SparkConf().setAppName("weightedAverage").setMaster("local[*]")
    val conf = new SparkConf().setAppName("Spark_MOPSO_Avg")
    val session = SparkSession.builder().config(conf).getOrCreate()
    val sc: SparkContext = session.sparkContext

    val initDataRDD: RDD[String] = sc.textFile(inputHDFS)

    // 处理原始数据
    val allDataWithKRDD: RDD[(Array[Double], Int)] = handleDataWithOutputDataWithK(initDataRDD,
      inputDataWithHeader = false,
      inputDataWithK = true,
      splitString = ",",
      labelStartWithZero)


    // 1. 按标签进行分区，
    //        val HashPartitionerValue = allDataWithKRDD.map { i => (i._2, i._1) }.partitionBy(new HashPartitioner(numberOfSubPop)).map { i => (i._2, i._1) } // 这个分区保证了每个分区只有一个簇
    //        val allDataWithKRDDRandomPartition: RDD[(Array[Double], Int)] = HashPartitionerValue // 再分区

    // 2.数据不均匀分区，即有的分区有几个簇，这里将分区数小于簇的数量
    //                val HashPartitionerValue = allDataWithKRDD.map { i => (i._2, i._1) }.partitionBy(new HashPartitioner(numberOfSubPop - 1)).map { i => (i._2, i._1) } // 这个分区保证了每个分区只有一个簇
    //                val allDataWithKRDDRandomPartition: RDD[(Array[Double], Int)] = HashPartitionerValue // 再分区

    //     3. 均匀分区
    var allDataWithKRDDRandomPartition = allDataWithKRDD.repartition(numberOfSubPop).persist() // 数据打乱分区,

    // numberOfClusters
    numberOfClusters = allDataWithKRDDRandomPartition.mapPartitions {
      i: Iterator[(Array[Double], Int)] =>
        val array = i.toArray
        array.map(_._2).distinct.toIterator
    }.distinct().collect().length

    //    numberOfClusters = 5
    // numberOfFeatures
    numberOfFeatures = allDataWithKRDDRandomPartition.map(_._1).first().length

    // 所有数据总量
    val allDataNumbers = allDataWithKRDDRandomPartition.mapPartitions {
      i =>
        Array(i.length).toIterator
    }.collect().sum

    // print base info
    val baseSetting = showBaseInfo(allDataWithKRDDRandomPartition,
      numberOfClusters,
      numberOfFeatures,
      numberOfMerge,
      inputHDFS,
      outputHDFS,
      numberOfSubPop,
      numberOfWorker,
      iter_max)

    // ======== all data in dataset ===============
    var inputDataValueRDD: RDD[Array[Double]] = allDataWithKRDDRandomPartition.map(_._1).persist()

    // 计算每个特征的上下限(max,min)
    val featureThreshold: Array[(Double, Double)] = getFeatureThreshold(inputDataValueRDD)
    featureThreshold.foreach(println)

    //  初始化粒子群的位置
    // 使用kmeans 算法来初始化种群的位置
    val KMeanCenter: Array[Array[Double]] = initPositionWithKmeans(inputDataValueRDD) // 种群位置x

    var initParticleSwarm = Array.range(0, numberOfParticles).map {
      index =>
        (initPositionWithDataPointMaxDistance(inputDataValueRDD, numberOfClusters) // 种群位置x
          , Array.fill[Double](numberOfClusters, numberOfFeatures)(Random.nextDouble() * (Vmax - Vmin) + Vmin) // 种群速度 v
          , Array.fill[Double](numberOfObjective)(0.0) // 当前的适应度值
          , Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 个体最优位置
          , Array.fill[Double](numberOfObjective)(0.0) // 个体最优值
          , Array.fill[Double](1)(0.0) // 拥挤度
        )
    }
    val initParticleSwarmBC = sc.broadcast(initParticleSwarm)

    val initParticleSwarmFitness = inputDataValueRDD.mapPartitionsWithIndex {
      (Index, Iterator) =>
        val partitionPoints = Iterator.toArray // 该分区的所有数据实例
        val partitionParticleSwarm = initParticleSwarmBC.value // 粒子群

        // 计算粒子群的适应度值
        val subParticleSwarm: Array[(Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])] = partitionParticleSwarm.map {
          iter =>
            val position: Array[Array[Double]] = iter._1
            val fitness = calFitnessNew(partitionPoints, position, allDataNumbers)
            Tuple6(iter._1, iter._2, fitness, iter._1, fitness, iter._6)
        }

        val map = mutable.Map[Int, Array[(Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])]]()

        map(Index) = subParticleSwarm
        map.toIterator
    }.collect()

    // 对每个分区的值进行相加
    var index = 0
    var particleSwarm = initParticleSwarm.map {
      i =>
        var one = 0.0
        var two = 0.0
        initParticleSwarmFitness.map {
          partitionIndex =>
            val value = partitionIndex._2
            val fitness = value(index)._3
            one = one + fitness(0)
            two = two + fitness(1)
        }
        val fitness = Array(one, two)
        index = 1 + index
        (i._1, i._2, fitness, i._1, fitness, i._6)
    }

    // 将粒子群中的粒子都复制到存档中，剔除支配解
    // Archive 组成 (位置，适应度值，拥挤度)
    var Archive = particleSwarm.map {
      line =>
        val Position: Array[Array[Double]] = line._1
        val Fitness: Array[Double] = line._3
        val distance: Array[Double] = line._6
        // 添加到Archive 中
        Tuple3(Position, Fitness, distance)
    }


    // 剔除支配解
    Archive = updateArchive(Archive)

    println()
    val initArchiveHead = Archive.map(_._2).head
    println("初始外部存档解的形状：" + Archive.map(_._2).length + "x" + initArchiveHead.length + ",初始外部存档的适应度值：")
    Archive.map(_._2).map(i => (i(0), i(1))).foreach(println)

    // 迭代操作
    while (iter <= iter_max) {
      // 更新W
      val w = (Wmax + iter * (Wmax - Wmin) * 1.0) / iter_max

      // 从 masterArchive 中获取一个全局最优解
      val globalBest = getGlobalBest(Archive)


      // 更新粒子
      // 更新粒子的速度、位置
      val updateParticleSwarm = updateParticleSwarmVelocityAndPosition(particleSwarm, globalBest, featureThreshold, w)
      // 更新粒子的适应度值、个体最优
      val updateParticleSwarmBC = sc.broadcast(updateParticleSwarm)

      val updateParticleSwarmFitness = inputDataValueRDD.mapPartitionsWithIndex {
        (Index, Iterator) =>
          val partitionPoints = Iterator.toArray // 该分区的所有数据实例
          val partitionParticleSwarm = updateParticleSwarmBC.value // 粒子群

          // 计算粒子群的适应度值
          val subParticleSwarm: Array[(Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])] = partitionParticleSwarm.map {
            iter =>
              val position: Array[Array[Double]] = iter._1
              val fitness = calFitnessNew(partitionPoints, position, allDataNumbers)
              Tuple6(iter._1, iter._2, fitness, iter._1, iter._5, iter._6)
          }

          val map = mutable.Map[Int, Array[(Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])]]()

          map(Index) = subParticleSwarm
          map.toIterator
      }.collect()

      var index = 0
      val updateParticleSwarmFitnessing = updateParticleSwarm.map {
        i =>
          var one = 0.0
          var two = 0.0
          updateParticleSwarmFitness.map {
            partitionIndex =>
              val value = partitionIndex._2
              val fitness = value(index)._3
              one = one + fitness(0)
              two = two + fitness(1)
          }
          val fitness = Array(one, two)
          index = 1 + index
          (i._1, i._2, fitness, i._4, i._5, i._6)
      }

      // 更新个体最优
      particleSwarm = updateParticleSwarmFitnessing.map {
        iter =>
          // 更新粒子的个体最优
          val BestPosition = iter._4
          val BestFitness = iter._5
          var newBestPosition: Array[Array[Double]] = Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 更新后的个体最优位置
          var newBestFitness = Array.fill(numberOfObjective)(0.0) // 更新后个体最优适应度值

          // 如果当前解优于（支配）当前局部最优副本则替换之；
          newBestFitness = BestFitness
          newBestPosition = BestPosition

          if (isDominatedBy(iter._3, BestFitness)) { //比较种群pop中第i个个体和个体i的pBest，若i比pBest优秀即i支配pBest则函数返回1
            // i支配pBest，更新pBest
            newBestFitness = iter._3
            newBestPosition = iter._1
          } else {
            // 若i和pBest无法比较，则随机选择
            // 两者互不支配，则随机决定是否用当前解替换局部最优解
            if (Random.nextDouble() < 0.5) {
              newBestFitness = iter._3
              newBestPosition = iter._1
            }
          }
          (iter._1, iter._2, iter._3, BestPosition, BestFitness, iter._6)
      }

      // 将粒子群中的粒子都复制到存档中，剔除支配解
      var newArchiveArrayBuffer = ArrayBuffer[(Array[Array[Double]], Array[Double], Array[Double])]()
      // 先将旧存档存入 newArchiveArrayBuffer
      Archive.foreach {
        i =>
          newArchiveArrayBuffer += i
      }
      // 再将粒子群中的粒子都复制到存档中
      particleSwarm.foreach {
        line =>
          val Position: Array[Array[Double]] = line._1
          val Fitness: Array[Double] = line._3
          val distance: Array[Double] = line._6
          // 添加到Archive 中
          newArchiveArrayBuffer += Tuple3(Position, Fitness, distance)
      }


      // 剔除支配解
      Archive = updateArchive(newArchiveArrayBuffer.toArray)


      println("第" + iter + "次迭代,Archive 后的大小" + Archive.size)

      updateParticleSwarmBC.destroy()

      iter = iter + 1
    }

    println()
    val endTime = System.nanoTime
    val duration = (endTime - start) / 1e9d
    println("Timer", duration)


    // 进行常规实验
    val allArchive = Archive

    allArchive.map {
      i =>
        println((i._2(0), i._2(1)))
    }

    // 数据归一化
    val archiveNormalization = dataNormalization(allArchive)

    // 对全部结果进行计算
    val allArchiveStringBuffer = ArrayBuffer[Array[String]]()
    val tuples1 = ArrayBuffer[(Array[Array[Double]], Array[Double], Array[Double])]()
    var number = 0
    allArchive.map {
      i =>
        var finalBestArchiveString = ArrayBuffer[String]()
        finalBestArchiveString += "==========new Archive============"
        val clusterCenter = i._1

        //        val DBIValue = DBI(allDataWithKRDDRandomPartition, clusterCenter)
        //        println("DBIValue: " + DBIValue)
        //        finalBestArchiveString += "DBIValue: " + DBIValue
        //
        //        val InertiaValue = calInertia(allDataWithKRDDRandomPartition, clusterCenter)
        //        println("InertiaValue: " + InertiaValue)
        //        finalBestArchiveString += "InertiaValue: " + InertiaValue

        val map = scala.collection.mutable.HashMap.empty[Int, Int]
        var calRealNum = 0

        for (j <- 1 until (numberOfClusters + 1)) {
          val real = allDataWithKRDDRandomPartition.filter(_._2 == j)
          val tuple = getTrueRateNew(real.map(_._1), clusterCenter)
          calRealNum += tuple._2
          map += (tuple._1 -> tuple._2)
          finalBestArchiveString += "簇：" + tuple._1 + ",占：" + tuple._2 + ",实际：" + tuple._3
        }
        val size = map.size
        if (size == numberOfClusters) {
          number = number + 1
          finalBestArchiveString += "正确的个数：" + calRealNum + ",准确率：" + (calRealNum * 1.0 / allDataNumbers)
          tuples1 += i
        }
        allArchiveStringBuffer += finalBestArchiveString.toArray
    }

    val allArchiveString: Array[Array[String]] = allArchiveStringBuffer.toArray


    println
    println("kmeans 的解集：")

    var kmeansArchiveString: ArrayBuffer[String] = ArrayBuffer[String]()
    //    val KmeansDBIValue = DBI(allDataWithKRDDRandomPartition, KMeanCenter)
    //    println("KmeansDBIValue: " + KmeansDBIValue)
    //    kmeansArchiveString += "KmeansDBIValue: " + KmeansDBIValue
    //
    //    val KmeansInertiaValue = calInertia(allDataWithKRDDRandomPartition, KMeanCenter)
    //    println("KmeansInertiaValue: " + KmeansInertiaValue)
    //    kmeansArchiveString += "KmeansInertiaValue: " + KmeansInertiaValue

    var calkmeansRealNum = 0
    var KmeansAccuracy = 0.0

    val kmeansmap = scala.collection.mutable.HashMap.empty[Int, Int]
    for (j <- 1 until (numberOfClusters + 1)) {
      val real = allDataWithKRDDRandomPartition.filter(_._2 == j)
      val tuple = getTrueRateNew(real.map(_._1), KMeanCenter)
      calkmeansRealNum += tuple._2
      kmeansmap += (tuple._1 -> tuple._2)
      kmeansArchiveString += "最好的簇：" + tuple._1 + ",占：" + tuple._2 + ",实际：" + tuple._3
    }
    val kmeanssize = kmeansmap.size
    if (kmeanssize == numberOfClusters) {
      KmeansAccuracy = (calkmeansRealNum * 1.0 / allDataNumbers)
      kmeansArchiveString += "正确的个数：" + calkmeansRealNum + ",准确率：" + KmeansAccuracy
    }
    kmeansArchiveString += "一共有" + number + "种方案可以选择。"


    showAndSaveArchive(sc,
      allArchive,
      archiveNormalization,
      KMeanCenter,
      inputHDFS,
      outputHDFS,
      duration,
      baseSetting,
      allArchiveString,
      kmeansArchiveString)
    session.stop()
  }


  // =============== 数据归一化 =================
  def dataNormalization(allArchive: Array[(Array[Array[Double]], Array[Double], Array[Double])]) = {
    // 进行数据归一化
    val array1: Array[Array[Double]] = allArchive.map(_._2)
    val x = array1.map(_ (0))
    val xMax = x.max
    val xMin = x.min
    val xD = xMax - xMin
    val y = array1.map(_ (1))
    val yMax = y.max
    val yMin = y.min
    val yD = yMax - yMin
    val archiveNormalization = allArchive.map {
      archice: (Array[Array[Double]], Array[Double], Array[Double]) =>
        val value = archice._2
        val doubles = ArrayBuffer[Double]()
        doubles += (value(0) - xMin) / xD
        doubles += (value(1) - yMin) / yD
        (archice._1, doubles.toArray, archice._3)
    }
    archiveNormalization
  }


  // ==========选取一个全局最优解 ===========
  def calAvgGlobalBest(ArchiveCost: Array[(Array[Array[Double]], Array[Double], Array[Double])]) = {
    val archiveLength = ArchiveCost.length

    val avgPosition = Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0)

    var one = 0.0
    var two = 0.0
    ArchiveCost.map {
      i =>
        val value: Array[Double] = i._2
        one += value(0)
        two += value(1)
        val position: Array[Array[Double]] = i._1
        for (clusterIndex <- 0 until (numberOfClusters)) {
          for (featureIndex <- 0 until (numberOfFeatures)) {
            avgPosition(clusterIndex)(featureIndex) += position(clusterIndex)(featureIndex)
          }
        }
    }
    var avgFitness = ArrayBuffer[Double]()
    avgFitness += (one * 1.0 / archiveLength)
    avgFitness += (two * 1.0 / archiveLength)

    // 计算粒子位置的均值
    for (clusterIndex <- 0 until (numberOfClusters)) {
      for (featureIndex <- 0 until (numberOfFeatures)) {
        avgPosition(clusterIndex)(featureIndex) = avgPosition(clusterIndex)(featureIndex) * 1.0 / archiveLength
      }
    }

    Tuple3(avgPosition, avgFitness.toArray, Array(0.0))
  }


  def getTrueRateNew(allDataValue: RDD[Array[Double]], mopsoCenters: Array[Array[Double]]) = {
    val allDataWithClusterK = allDataValue.map {
      line =>
        val distanceArray: ArrayBuffer[(Int, Double)] = new ArrayBuffer[(Int, Double)]() // 表示：（Int,Distance） ==> 哪个簇，距离
        var k = 1
        for (elem <- mopsoCenters) {
          //          val distance = weightedEuclideanDistance(line, elem)
          val distance = dist(line, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK = distanceArray.minBy(_._2)
        (lineK._1, line)
    }
    val value = allDataWithClusterK.map {
      i =>
        val k = i._1
        (k, 1)
    }

    val result = value.reduceByKey((a, b) => (a + b)).collect()
    val allDataSum = result.map(_._2).sum

    val maxCount: (Int, Int) = result.maxBy(_._2)
    Tuple3(maxCount._1, maxCount._2, allDataSum)
    //    value.foreach(println)
    //    value.repartition(1).saveAsTextFile(outputHDFS + "\\TrueRate\\" + NowDate())
  }


  // ===================== 更新粒子群的位置、速度 =======================================
  def updateParticleSwarmVelocityAndPosition(particleSwarmFitness: Array[(Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])],
                                             globalBest: (Array[Array[Double]], Array[Double], Array[Double]),
                                             featureThreshold: Array[(Double, Double)],
                                             w: Double) = {
    val c1 = 1 // Personal Learning Coefficient 个人学习因子
    val c2 = 2 // Global Learning Coefficient 全局学习因子

    var Vmax = 1.0
    var Vmin = -1.0

    val globalBestPosition = globalBest._1

    val updateParticleSwarmWithVelocityAndPosition = particleSwarmFitness.map {
      particle =>
        val position = particle._1
        val velocity = particle._2
        val BestPosition = particle._4

        val newPosition: Array[Array[Double]] = Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 更新后的位置
        val newVelocity: Array[Array[Double]] = Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 更新后的速度

        // 粒子速度和位置更新
        List.range(0, numberOfClusters).foreach {
          i =>
            List.range(0, numberOfFeatures).foreach {
              j =>
                // 更新速度信息
                newVelocity(i)(j) = w * velocity(i)(j) +
                  c1 * Random.nextDouble() * (BestPosition(i)(j) - position(i)(j)) +
                  c2 * Random.nextDouble() * (globalBestPosition(i)(j) - position(i)(j))

                if (newVelocity(i)(j) > Vmax || newVelocity(i)(j) < Vmin) {
                  newVelocity(i)(j) = Random.nextDouble() * (Vmax - Vmin) + Vmin
                }

                // 更新位置信息
                newPosition(i)(j) = position(i)(j) + newVelocity(i)(j)
            }
        }

        (newPosition, newVelocity, particle._3, particle._4, particle._5, particle._6)
    }
    updateParticleSwarmWithVelocityAndPosition
  }

  // =============== 计算准确率 ===============
  def getAccuracyRate(allDataWithTrueLabelAndCalLabel: RDD[(Int, Int)]) = {
    var trueCount: Long = 0

    trueCount = allDataWithTrueLabelAndCalLabel.filter {
      i =>
        i._1 == i._2
    }.collect().length


    val allDataCount = allDataWithTrueLabelAndCalLabel.collect().length
    (trueCount * 1.0) / allDataCount
  }


  // =============== 根据欧几里得距离将数据集中的点指定给最近的质心 ===================================
  def calTrueKWithCalKKmeans(allDataValue: Iterator[(Array[Double], Int)], clusterCenter: Array[Array[Double]]) = {
    allDataValue.map {
      line =>
        // 表示：（Int,Distance） ==> 哪个簇，距离
        val distanceArray = new ArrayBuffer[(Int, Double)]()
        var k = 1
        for (elem <- clusterCenter) {
          //          val distance = weightedEuclideanDistance(line._1, elem)
          val distance = dist(line._1, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK: (Int, Double) = distanceArray.minBy(_._2)
        (line._2, lineK._1)
    }
  }


  // ============== 根据轮廓系数从Archive 中选择一个最佳的解作为最后的解==============
  def selectBestArchiveAsFinalResult(Archive: Array[(Array[Array[Double]], Array[Double], Array[Double])],
                                     inputData: RDD[(Array[Double], Int)]) = {
    val ArchiveCenters: Array[Array[Array[Double]]] = Archive.map(_._1)
    val silhouetteCoefficientArray = ArrayBuffer[Double]()

    val allCenterSilhouetteCoefficientSum: Array[(Array[Array[Double]], Double)] = ArchiveCenters.map {
      centers: Array[Array[Double]] =>

        val silhouetteCoefficientSumMean = silhouetteCoefficient(inputData, centers, numberOfClusters)

        println("轮廓系数为：" + silhouetteCoefficientSumMean)
        silhouetteCoefficientArray += silhouetteCoefficientSumMean

        (centers, silhouetteCoefficientSumMean)
    }


    val theBestArchive = allCenterSilhouetteCoefficientSum.maxBy(_._2)
    val finalBestArchive = theBestArchive._1
    // 升序
    val AscendingOrder = finalBestArchive.sortBy(i => i(0))
    // 降序
    //    val DescendingOrder = AscendingOrder.reverse
    (silhouetteCoefficientArray.toArray, AscendingOrder)
  }


  // ================= 轮廓系数均值 ============
  def silhouetteCoefficient(inputData: RDD[(Array[Double], Int)], center: Array[Array[Double]], numberOfClusters: Int) = {
    val sc = inputData.sparkContext

    val inputDataWithPoints: RDD[Array[Double]] = inputData.map(_._1) cache()
    val clusterCenterBC = sc.broadcast(center)

    val allPointWithCalK = inputDataWithPoints.mapPartitions {
      i =>
        calPartitionKmeans(i, clusterCenterBC.value)
    }

    val allPointWithKRDD: RDD[(Int, Array[Double])] = allPointWithCalK.map(i => (i._1._1, i._2))
    val allPointWithKArray = allPointWithKRDD.collect()
    val length = allPointWithKArray.length

    var S = 0.0
    for (i <- 0 until length) {
      val a = allPointWithKArray(i)
      val aArray = allPointWithKArray.filter(_._1 == a._1)
      var asum = 0.0
      var alength = 0
      aArray.foreach {
        i =>
          //          asum += weightedEuclideanDistance(i._2, a._2)
          asum += dist(i._2, a._2)
          alength = alength + 1
      }
      val ai = asum / alength

      var bi = Double.MaxValue
      for (k <- 1 until (numberOfClusters + 1)) {
        if (k != a._1) {
          val bArray = allPointWithKArray.filter(_._1 == k)
          var bsum = 0.0
          var blength = 0
          bArray.foreach {
            i =>
              //              bsum += weightedEuclideanDistance(i._2, a._2)
              bsum += dist(i._2, a._2)
              blength = blength + 1
          }
          val biTemp = bsum / blength
          bi = Math.min(bi, biTemp)
        }
      }
      val si = (bi - ai) / Math.max(ai, bi)
      S += si
    }
    val sMean = S / length
    sMean
  }


  // =============== 根据欧几里得距离将数据集中的点指定给最近的质心 ===================================
  def calPartitionKmeans(allDataValue: Iterator[Array[Double]], clusterCenter: Array[Array[Double]]) = {
    allDataValue.map {
      line =>
        val distanceArray: ArrayBuffer[(Int, Double)] = new ArrayBuffer[(Int, Double)]() // 表示：（Int,Distance） ==> 哪个簇，距离
        var k = 1
        for (elem <- clusterCenter) {
          //          val distance = weightedEuclideanDistance(line, elem)
          val distance = dist(line, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK = distanceArray.minBy(_._2)
        (lineK, line)
    }
  }


  // ======== 输出到控制台并保存 ========================
  def showAndSaveArchive(sc: SparkContext,
                         Archive: Array[(Array[Array[Double]], Array[Double], Array[Double])],
                         archiveNormalization: Array[(Array[Array[Double]], Array[Double], Array[Double])],
                         KmeansPosition: Array[Array[Double]],
                         DataSetPath: String,
                         savePath: String,
                         duration: Double,
                         baseSetting: String,
                         allArchiveString: Array[Array[String]],
                         kmeansArchiveString: ArrayBuffer[String]
                        ): Unit = {
    val MOPSOOUTPUTAndKmeans = new ArrayBuffer[String]()
    MOPSOOUTPUTAndKmeans += baseSetting + "\n"
    MOPSOOUTPUTAndKmeans += "数据集：" + DataSetPath + "\n"
    val outputPath = savePath + "/" + NowDate()
    MOPSOOUTPUTAndKmeans += "结果存档：" + outputPath + "\n"


    // 打印聚类效果评价指标
    MOPSOOUTPUTAndKmeans += "运行时间为：" + duration + "\n"


    println()
    val head = Archive.map(_._2).head
    println("外部存档解的形状：" + Archive.map(_._2).length + "x" + head.length + ",外部存档的适应度值：")
    MOPSOOUTPUTAndKmeans += "外部存档解的形状：" + Archive.map(_._2).length + "x" + head.length + ",外部存档的适应度值："
    Archive.map(_._2).map(i => (i(0), i(1))).foreach(println)
    // 保存适应度值
    Archive.map(i => (i._2(0), i._2(1))).foreach {
      i =>
        val str = i._1 + "," + i._2
        MOPSOOUTPUTAndKmeans += str
    }
    MOPSOOUTPUTAndKmeans += "\n"


    println()
    val archiveNormalizationHead = archiveNormalization.map(_._2).head
    println("归一化后外部存档解的形状：" + archiveNormalization.map(_._2).length + "x" + archiveNormalizationHead.length + ",外部存档的适应度值：")
    MOPSOOUTPUTAndKmeans += "归一化后外部存档解的形状：" + archiveNormalization.map(_._2).length + "x" + archiveNormalizationHead.length + ",外部存档的适应度值："
    archiveNormalization.map(_._2).map(i => (i(0), i(1))).foreach(println)
    // 保存适应度值
    archiveNormalization.map(i => (i._2(0), i._2(1))).foreach {
      i =>
        val str = i._1 + "," + i._2
        MOPSOOUTPUTAndKmeans += str
    }

    MOPSOOUTPUTAndKmeans += "\n"

    // 输出簇中心的位置
    println()
    for (position: Array[Array[Double]] <- Archive.map(_._1)) {
      var clusterIndex: Int = 0
      for (clusterKPosition: Array[Double] <- position) {
        var s = ""
        for (elem <- clusterKPosition) {
          s += elem + ","
        }
        // 删除最后的逗号
        s = s.substring(0, s.length() - 1)
        println("MOPSO Center Point of Cluster " + (clusterIndex + 1) + "==》  " + s)
        MOPSOOUTPUTAndKmeans += "MOPSO Center Point of Cluster " + (clusterIndex + 1) + "==》  " + s
        clusterIndex += 1
      }
      MOPSOOUTPUTAndKmeans += "\n"
      println()
    }

    KmeansPosition.foreach {
      var clusterIndex: Int = 0
      cluster: Array[Double] =>
        var lineString = ""
        cluster.foreach {
          index =>
            lineString += index + ","
        }
        // 删除最后的逗号
        lineString = lineString.substring(0, lineString.length() - 1)
        println("Kmeans Center Point of Cluster " + (clusterIndex + 1) + "==》  " + lineString)
        MOPSOOUTPUTAndKmeans += "Kmeans Center Point of Cluster " + (clusterIndex + 1) + "==》  " + lineString
        clusterIndex += 1
    }

    println()


    println()
    MOPSOOUTPUTAndKmeans += "\n"
    MOPSOOUTPUTAndKmeans += "所有存档的分布情况"
    println("所有存档的分布情况")
    var index = 1
    allArchiveString.map {
      j =>

        j.foreach {
          i =>
            println(i)
            MOPSOOUTPUTAndKmeans += i
        }

        println(index)
        println()
        MOPSOOUTPUTAndKmeans += "\n"
        index = index + 1
    }

    MOPSOOUTPUTAndKmeans += "\n"
    MOPSOOUTPUTAndKmeans += "kmeans 的分布情况"
    println("\nkmeans 的分布情况")
    kmeansArchiveString.foreach {
      i =>
        println(i)
        MOPSOOUTPUTAndKmeans += i
    }

    println("\n外部存档解的形状：" + Archive.map(_._2).length)

    sc.parallelize(MOPSOOUTPUTAndKmeans, 1).saveAsTextFile(outputPath)

  }

  // ========================= 获取当前时间 =====================
  def NowDate(): String = {
    val now: Date = new Date()
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss")
    val date = dateFormat.format(now)
    date
  }


  // =============== 对所有分区的适应度进行计算拥挤度 ===============
  def calculatePartitionCrowdingDistance(fitnesss: Array[Array[Double]]): Array[(Array[Double], Array[Double])] = {
    val sort: Array[Array[Double]] = fitnesss.sortBy(_ (1))
    val sortfronts: Array[(Array[Double], Array[Double])] = sort.map {
      i =>
        (i, Array.fill[Double](1)(0.0))
    }
    val size = sortfronts.size
    if (size == 1) {
      sortfronts(0)._2(0) = Double.PositiveInfinity
      return sortfronts
    }
    if (size == 2) {
      sortfronts(0)._2(0) = Double.PositiveInfinity
      sortfronts(0)._2(0) = Double.PositiveInfinity
      return sortfronts
    }

    // 端点拥挤距离设为极大值
    sortfronts(0)._2(0) == Double.PositiveInfinity
    sortfronts(size - 1)._2(0) = Double.PositiveInfinity
    // 为中间的各解计算拥挤距离
    for (i <- 1 until size - 1) {
      val cur = sortfronts(i)
      val pre = sortfronts(i - 1)
      val next = sortfronts(i + 1)
      val d = Math.abs(pre._1(0) - next._1(0)) * Math.abs(pre._1(1) - next._1(1))
      cur._2(0) = d
    }
    sortfronts
  }

  // =========== 计算适应度值 ==============
  def calFitnessNew(partitionData: Array[Array[Double]], positions: Array[Array[Double]], allDataNumbers: Int) = {
    val length = partitionData.length
    val rate = length * 1.0 / allDataNumbers
    //    val rate = 1
    val allDataWithClusterK = partitionData.map {
      line =>
        val distanceArray: ArrayBuffer[(Int, Double)] = new ArrayBuffer[(Int, Double)]() // 表示：（Int,Distance） ==> 哪个簇，距离
        var k = 1
        for (elem <- positions) {
          //          val distance = weightedEuclideanDistance(line, elem)
          val distance = dist(line, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK = distanceArray.minBy(_._2)
        (lineK, line)
    }
    val DevValue = Dev(allDataWithClusterK) * rate
    val ConnValue = Conn(allDataWithClusterK) * rate
    val fitness: Array[Double] = Array(DevValue, ConnValue)
    fitness
  }


  // ===================== 更新Archive ========================
  // (位置，适应度值，拥挤度)
  def updateArchive(ArchiveCost: Array[(Array[Array[Double]], Array[Double], Array[Double])]) = {
    // 计算支配情况
    // (位置，适应度值，拥挤度,支配情况)
    val dominatedState: Array[(Array[Array[Double]], Array[Double], Array[Double], Boolean)] = ArchiveCost.map {
      line =>
        // (位置，适应度值，拥挤度)
        val cost: Array[Double] = line._2
        val bool = isDominatedIn(cost, ArchiveCost)
        (line._1, line._2, line._3, bool)
    }

    // (位置，适应度值，拥挤度)
    // 这里删除支配解，保留非支配解，即只保留false
    val updateArchiveIngArrayBuffer = ArrayBuffer[(Array[Array[Double]], Array[Double], Array[Double])]()
    dominatedState.foreach {
      i =>
        if (i._4 == false) {
          updateArchiveIngArrayBuffer += Tuple3(i._1, i._2, i._3)
        }
    }
    var updateArchiveIng: Array[(Array[Array[Double]], Array[Double], Array[Double])] = updateArchiveIngArrayBuffer.toArray

    val overflow = updateArchiveIng.size - repository
    // 如果存档集合溢出则截断之（剔除最拥挤的粒子）
    // 拥挤距离算法：如果某个解周围的解集密度很大，那么其中一些解就会被删除，这样可以减少很多不必要的计算，加快算法的运行速度
    if (overflow > 0) {
      updateArchiveIng = calculateCrowdingDistance(updateArchiveIng)
      updateArchiveIng = updateArchiveIng.sortBy(_._3(0)) // 根据拥挤度小到大排序
      updateArchiveIng = updateArchiveIng.drop(overflow)
    }
    updateArchiveIng
  }

  // ==========选取一个全局最优解。此方法会检查存档集合粒子的拥挤距离并返回拥挤度最低的个体===========
  // ArchiveCost (位置，适应度值，拥挤度)
  def getGlobalBest(ArchiveCost: Array[(Array[Array[Double]], Array[Double], Array[Double])]): (Array[Array[Double]], Array[Double], Array[Double]) = {
    val archiveCrowdingDistance: Array[(Array[Array[Double]], Array[Double], Array[Double])] = calculateCrowdingDistance(ArchiveCost)
    val size = archiveCrowdingDistance.size
    if (size == 1) {
      val result: (Array[Array[Double]], Array[Double], Array[Double]) = archiveCrowdingDistance(0)
      return result
    }
    if (size == 2) {
      val i = Math.abs(new Random().nextInt(2))
      val result = archiveCrowdingDistance(i)
      return result
    }

    // 计算最大拥挤距离
    val archiveCrowdingDistanceRemoveFirst = archiveCrowdingDistance.drop(1) // 避开值为无穷大的两个端点）
    val archiveCrowdingDistanceRemoveFirstAndRemoveLast = archiveCrowdingDistanceRemoveFirst.dropRight(1) // 避开值为无穷大的两个端点）
    val maxCrowdingDistance: Double = archiveCrowdingDistanceRemoveFirstAndRemoveLast.maxBy(_._3(0))._3(0) // 找到拥挤度最大的

    // 获取拥挤距离最大的子集（避开值为无穷大的两个端点）
    // 找到拥挤度最大的子集
    val leastCrowdedArrayBuffer = ArrayBuffer[(Array[Array[Double]], Array[Double], Array[Double])]()
    archiveCrowdingDistance.foreach {
      i =>
        if (i._3(0) == maxCrowdingDistance) {
          leastCrowdedArrayBuffer += i
        }
    }
    val leastCrowded = leastCrowdedArrayBuffer.toArray
    // 从子集中随机返回一个粒子
    val index = Random.nextInt(leastCrowded.size)
    val result = leastCrowded(index)

    result
  }


  // ==========选取一个全局最优解。此方法会检查存档集合粒子的拥挤距离并返回拥挤度最低的个体===========
  // ArchiveCost (位置，适应度值，拥挤度)
  def getGlobalBestNew(ArchiveCost: Array[(Array[Array[Double]], Array[Double], Array[Double])]): (Array[Array[Double]], Array[Double], Array[Double]) = {
    val archiveCrowdingDistance: Array[(Array[Array[Double]], Array[Double], Array[Double])] = calculateCrowdingDistance(ArchiveCost)
    val size = archiveCrowdingDistance.size
    if (size == 1) {
      val result: (Array[Array[Double]], Array[Double], Array[Double]) = archiveCrowdingDistance(0)
      return result
    }
    if (size == 2) {
      val i = Math.abs(new Random().nextInt(2))
      val result = archiveCrowdingDistance(i)
      return result
    }

    // 计算最大拥挤距离
    val archiveCrowdingDistanceRemoveFirst = archiveCrowdingDistance.drop(1) // 避开值为无穷大的两个端点）
    val archiveCrowdingDistanceRemoveFirstAndRemoveLast = archiveCrowdingDistanceRemoveFirst.dropRight(1) // 避开值为无穷大的两个端点）
    val sortArchive: Array[(Array[Array[Double]], Array[Double], Array[Double])] = archiveCrowdingDistanceRemoveFirstAndRemoveLast.sortBy(_._3(0))
    val top: Int = (sortArchive.length * 0.1).toInt + 1
    sortArchive(Random.nextInt(top))
  }

  // ===================计算前沿解集中解的拥挤距离。=======================
  // (位置，适应度值，拥挤度)
  def calculateCrowdingDistance(updateArchiveIng: Array[(Array[Array[Double]], Array[Double], Array[Double])]): Array[(Array[Array[Double]], Array[Double], Array[Double])] = {
    val sortfronts: Array[(Array[Array[Double]], Array[Double], Array[Double])] = sortFronts(updateArchiveIng)

    val size = sortfronts.size
    if (size == 1) {
      sortfronts(0)._3(0) = Double.PositiveInfinity
      return sortfronts
    }
    if (size == 2) {
      sortfronts(0)._3(0) = Double.PositiveInfinity
      sortfronts(0)._3(0) = Double.PositiveInfinity
      return sortfronts
    }
    // 端点拥挤距离设为极大值
    sortfronts(0)._3(0) == Double.PositiveInfinity
    sortfronts(size - 1)._3(0) = Double.PositiveInfinity

    // 为中间的各解计算拥挤距离
    for (i <- 1 until size - 1) {
      val cur = sortfronts(i)
      val pre = sortfronts(i - 1)
      val next = sortfronts(i + 1)

      val d1 = math.sqrt(dist(pre._2, cur._2))
      val d2 = math.sqrt(dist(next._2, cur._2))
      val d = (d1 + d2) / 2.0
      cur._3(0) = d
    }
    sortfronts
  }


  // ========================对前沿解集进行排序（按F2降序）。==================
  // (位置，适应度值，拥挤度)
  def sortFronts(updateArchiveIng: Array[(Array[Array[Double]], Array[Double], Array[Double])]) = {
    val sort = updateArchiveIng.sortBy(_._2(1))
    sort
  }


  // ======================检查该解在指定集合中是否为受控解（次优解）。=====================
  // Archive ==> (位置，适应度值，拥挤度)
  def isDominatedIn(cost: Array[Double], Archive: Array[(Array[Array[Double]], Array[Double], Array[Double])]): Boolean = {
    for (elem <- Archive) {
      val archiveCost: Array[Double] = elem._2
      val bool = isDominatedBy(cost, archiveCost)
      if (bool) {
        return true
      }
    }
    false
  }

  // ========================判断当前解是否劣于另一解。 ===========================
  def isDominatedBy(costOne: Array[Double], costTwo: Array[Double]): Boolean = {
    val costOneF1 = costOne(0)
    val costOneF2 = costOne(1)
    val costTwoF1 = costTwo(0)
    val costTwoF2 = costTwo(1)
    costTwoF1 <= costOneF1 && costTwoF2 < costOneF2 || costTwoF1 < costOneF1 && costTwoF2 < costOneF2
  }


  // =================== 适应性函数 聚类连通性======================
  def Dev(allDataWithClusterK: Array[((Int, Double), Array[Double])]) = {
    val result = allDataWithClusterK.map(_._1._2).sum
    result
  }

  // =================== 适应性函数 聚类连通性======================
  def Conn(allDataWithClusterK: Array[((Int, Double), Array[Double])]) = {
    var result = 0.0
    val length = allDataWithClusterK.length

    for (i <- 0 until (length)) {
      var distances = new ArrayBuffer[(Int, Double)]() // (Int, Double)==>(簇k,distance)
      for (j <- 0 until (length)) {
        if (i != j) {
          // 计算最近邻
          val distance = dist(allDataWithClusterK(i)._2, allDataWithClusterK(j)._2)
          distances += Tuple2(allDataWithClusterK(j)._1._1, distance)
        }
      }
      // 排序
      val distanceSorted = distances.sortBy(_._2)
      // 获得 L 个最近邻
      val nn: ArrayBuffer[(Int, Double)] = distanceSorted.take(LIndex)

      // 计算Xij 的值
      for (index <- 0 until (LIndex)) {
        // 判断Xi 与Xj 是否是同一个类
        if (allDataWithClusterK(i)._1._1 == nn(index)._1) {
          result += 1.0 / (index + 1)
        }
      }
    }
    result
  }


  // ================= 求解欧几里得距离 =================
  def dist(x: Array[Double], y: Array[Double]) = {
    var sum = 0.0
    for (i <- 0 until x.length) {
      sum += Math.pow((x(i) - y(i)), 2)
    }
    Math.sqrt(sum)
    //    val  d = weightedEuclideanDistance(x,y)
    //    d
  }


  def calWeighted(x: Array[Double]) = {
    val xSum: Double = x.sum
    val weight = x.map {
      i =>
        i * 1.0 / xSum
    }
    weight
  }

  // ====================== 加权欧几里得距离 ===================
  def weightedEuclideanDistance(x: Array[Double], centers: Array[Double]) = {
    var sum = 0.0
    val xSum = x.sum
    for (i <- 0 until x.length) {
      val d = Math.pow((x(i) - centers(i)), 2) * (x(i) * 1.0 / xSum)
      sum += d
    }
    Math.sqrt(sum)
  }

  // =============== 初始化粒子的位置 ,返回的大小为：numberOfClusters * numberOfFeatures =============
  def initPositionWithKmeans(input: RDD[Array[Double]]) = {
    val result: Array[Array[Double]] = Array.ofDim[Double](numberOfClusters, numberOfFeatures)
    val parsedTrainingData: RDD[linalg.Vector] = input.map {
      line =>
        Vectors.dense(line)
    }.cache()
    val clusters: KMeansModel = KMeans.train(parsedTrainingData, numberOfClusters, numberOfKMeansIter)
    var index = 0
    clusters.clusterCenters.foreach {
      line =>
        result(index) = line.toArray
        index = index + 1
    }
    // 升序
    val AscendingOrder = result.sortBy(i => i(0))
    // 降序
    //    val DescendingOrder = AscendingOrder.reverse
    //    DescendingOrder
    AscendingOrder
    //    result
  }


  // ====================== 初始化粒子群，随机从数据中选取 ==========================
  def initPositionWithDataPointMaxDistance(input: RDD[Array[Double]], numberOfCluster: Int) = {
    val sc = input.sparkContext
    val initCentersArrayBuffer = ArrayBuffer[Array[Double]]()
    // 随机一个初始的聚类中心
    val firstCenter: Array[Double] = input.takeSample(false, 1)(0)
    initCentersArrayBuffer += firstCenter

    //    println("firstCenter:" + firstCenter.toList)

    // 先求第二个聚类中心
    val firstCenterBC = sc.broadcast(firstCenter)
    val firstDistance: RDD[(Double, Array[Double])] = input.mapPartitions {
      i =>
        val firstCenterBCValue = firstCenterBC.value
        val array = i.toArray // 该分区的所有数据
        // 计算该分区的所有数据到firstcenter 的距离
        val dataToFirstCenter: Array[(Double, Array[Double])] = array.map {
          j =>
            //            val d: Double = weightedEuclideanDistance(firstCenterBCValue, j)
            val d: Double = dist(firstCenterBCValue, j)
            (d, j)
        }
        val MaxDistanceInThisPartition = dataToFirstCenter.maxBy(_._1) // firstCenter 到该分区的最大距离
        Array(MaxDistanceInThisPartition).toIterator
    }
    val secondCenter: Array[Double] = firstDistance.collect().maxBy(_._1)._2

    //    println("secondCenter:" + secondCenter.toList)
    firstCenterBC.destroy()
    initCentersArrayBuffer += secondCenter

    // 三个点及以后，
    // 然后再选择距离前两个点的最短距离最大的那个点作为第三个初始类簇的中心点
    for (i <- 2 until numberOfCluster) {
      val initCentersArrayBufferBC = sc.broadcast(initCentersArrayBuffer.toArray)

      val otherDistance = input.mapPartitions {
        i =>
          val initCentersArrayBufferBCValue: Array[Array[Double]] = initCentersArrayBufferBC.value
          val array = i.toArray // 该分区的所有数据

          val value: (Double, Array[Double]) = array.map {
            j =>
              val jToCenter: Array[(Double, Array[Double])] = initCentersArrayBufferBCValue.map {
                centerI =>
                  //                  val d = weightedEuclideanDistance(centerI, j)
                  val d = dist(centerI, j)
                  (d, centerI)
              }
              val tuple: (Double, Array[Double]) = jToCenter.minBy(_._1)
              (tuple._1, j)
          }.maxBy(_._1)
          Array(value).toIterator

      }

      val otherCenter: Array[Double] = otherDistance.collect().maxBy(_._1)._2
      initCentersArrayBuffer += otherCenter

      //      println((i + 1) + ":Center" + otherCenter.toList)
      initCentersArrayBufferBC.destroy()
    }

    initCentersArrayBuffer.toArray.sortBy(i => i(0))
  }


  // ===================== 处理数据，输出数据带k========================================
  def handleDataWithOutputDataWithK(input: RDD[String], inputDataWithHeader: Boolean, inputDataWithK: Boolean, splitString: String, labelStartWithZero: Int): RDD[(Array[Double], Int)] = {
    var inputData: RDD[String] = input
    val sc = inputData.sparkContext

    // 广播
    val labelStartWithZeroBC: Broadcast[Int] = sc.broadcast(labelStartWithZero)

    // 如果你的csv文件有标题 的话，需要剔除首行
    if (inputDataWithHeader) { // 数据文件有标题 的话，需要剔除首行
      val header = input.first() //第一行 print(header)
      inputData = input.filter(i => i != header) //删除第一行
    }


    if (inputDataWithK) { // 输入的数据带有K值
      val result = inputData.map {
        line =>
          val fields: Array[String] = line.split(splitString)
          val doubles = new ArrayBuffer[Double]()
          for (i <- 0 until (fields.length - 1)) {
            doubles += fields(i).toDouble
          }
          val labelStartWithZeroBCValue = labelStartWithZeroBC.value
          if (labelStartWithZeroBCValue == 1) {
            // 说明label 是从1 开始
            (doubles.toArray, fields(fields.length - 1).toInt)
          } else {
            // 说明label 是从0 开始
            (doubles.toArray, fields(fields.length - 1).toInt + 1)
          }

      }
      return result
    } else { // 输入的数据不带有K值
      val result = inputData.map {
        line =>
          val doubles: Array[Double] = line.split(splitString)
            .map(_.trim)
            .filter(!"".equals(_))
            .map(_.toDouble)
          (doubles, 0)

      }
      return result
    }
  }

  // ================== 得到每个特征的上下限 ================================
  def getFeatureThreshold(inputDataValueRDD: RDD[Array[Double]]) = {
    val tuples = ArrayBuffer[(Double, Double)]()

    for (i <- 0 until (numberOfFeatures)) {
      val featureI: RDD[Double] = inputDataValueRDD.map {
        j =>
          j(i)
      }
      val min = featureI.min()
      val max: Double = featureI.max()
      tuples += Tuple2(max, min)
    }
    tuples.toArray
  }

  //  =============================== 显示基础信息 =============================
  //     showBaseInfo(inputData,numberOfClusters,numberOfFeatures,inputHDFS,outputHDFS)
  def showBaseInfo(inputData: RDD[(Array[Double], Int)],
                   numberOfClustersInput: Int,
                   numberOfFeaturesInput: Int,
                   numberOfMergeInput: Int,
                   inputHDFSInput: String,
                   outputHDFSInput: String,
                   numberOfSubPop: Int,
                   numberOfWorker: Int,
                   iter_max: Int) = {
    println("********* How many cluster are exist in each RDD partition ***********")
    var baseCluster = ""
    inputData.mapPartitionsWithIndex {
      (partId, iter) => {
        var map = mutable.Map[String, List[String]]()
        var part_name: String = "分区_" + partId
        map(part_name) = List[String]()
        while (iter.hasNext) {
          val next: (Array[Double], Int) = iter.next()
          var lineString = "clusterK_" + next._2
          map(part_name) :+= lineString // :+= 集合追加
        }
        map.iterator
      }
    }.map {
      i =>
        val value2: List[String] = i._2
        var count = 0
        for (k <- 1 until (numberOfClusters + 1)) {
          val str = "clusterK_" + k
          if (value2.contains(str)) {
            count = count + 1
          }
        }
        baseCluster += i._1 + ",有" + count + "个簇。\n"
        println(i._1 + ",有" + count + "个簇。")
    }.collect()

    // 基础参数设置
    var baseSetting = baseCluster
    val modelSetting = "模型：" + this.getClass.getName + ",\n粒子的个数：" + numberOfParticles + ", 存档大小:" + repository + ", k 值：" + numberOfClustersInput + ",特征数：" +
      numberOfFeaturesInput + ",最近邻: " + LIndex + ", 迭代次数：" +
      iter_max + ", 并行度：" + numberOfWorker + ",分区数：" + numberOfSubPop + ", 相隔几代合并种群：" + numberOfMergeInput +
      "\n输入的数据：" + inputHDFSInput + "\n结果保存路径：" + outputHDFSInput
    baseSetting += modelSetting
    println(baseSetting)
    logInfo("基础参数：" + baseSetting)
    baseSetting
  }

  // ================= Davies-Bouldin Index(戴维森堡丁指数)(分类适确性指标)(DB) ============
  def DBI(inputData: RDD[(Array[Double], Int)], clusterCenter: Array[Array[Double]]) = {
    val sc = inputData.sparkContext
    val length = clusterCenter.length
    val inputDataWithPoints: RDD[Array[Double]] = inputData.map(_._1) cache()
    val clusterCenterBC = sc.broadcast(clusterCenter)

    val value0: RDD[((Int, Double), Array[Double])] = inputDataWithPoints.mapPartitions {
      i =>
        calPartitionKmeans(i, clusterCenterBC.value)
    }
    val value: RDD[(Int, Double)] = value0.map(_._1)
    var max = Double.MinValue
    var result = 0.0

    val value3: RDD[(Int, Double)] = value0.map {
      i =>
        (i._1._1, (i._1._2, 1))
    }.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .map(t => (t._1, t._2._1 / t._2._2))
    val withinClusterDistance: Array[(Int, Double)] = value3.collect()
    for (i <- 0 until (length)) {
      val si = withinClusterDistance.filter(_._1 == (i + 1))(0)
      for (j <- i until (length)) {
        if (i != j) {
          val sj = withinClusterDistance.filter(_._1 == (j + 1))(0)
          //          val dij = weightedEuclideanDistance(clusterCenter(i), clusterCenter(j))
          val dij = dist(clusterCenter(i), clusterCenter(j))
          val Rij = (si._2 + sj._2) / dij
          if (Rij > max) {
            max = Rij
          }
        }
      }
      result = result + max
    }
    result / length
  }


  // ========================== 簇内平方和 ========================
  // 正确
  /*  这个公式被称为簇内平⽅和(cluster Sum of Square)，⼜叫做Inertia，统计学中 “和⽅差”、“簇内离差平⽅和”（SSE）在这⾥
      指同⼀意思。⽽将⼀个数据集中的所有簇的簇内平⽅和相加，就得到了整体平⽅和(Total Cluster Sum of Square)，⼜叫做total
      inertia，TSSE。Total Inertia越⼩，代表着每个簇内样本越相似，聚类的效果就越好。因此KMeans追求的是，求解能够让Inertia最
      ⼩化的质⼼。*/
  def calInertia(inputData: RDD[(Array[Double], Int)], center: Array[Array[Double]]) = {
    val sc = inputData.sparkContext
    val inputDataWithPoints: RDD[Array[Double]] = inputData.map(_._1) cache()
    val clusterCenterBC = sc.broadcast(center)

    val result = inputDataWithPoints.mapPartitions {
      i =>
        calPartitionKmeans(i, clusterCenterBC.value)
    }.map {
      i => (i._1._2 * i._1._2)
    }.sum()

    result
  }

}
