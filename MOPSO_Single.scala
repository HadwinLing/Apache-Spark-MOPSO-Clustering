import java.io._
import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random

/**
 * @Classname SparkMOPSOClusteringDataParallelism
 * @Description TODO
 * @Date 2022/5/1 14:58
 * @Created by ${hadwinling}
 */
object SingleMOPSOClustering {
  var iter = 1 // 循环计数器
  val c1 = 1.42
  val c2 = 1.63
  val Wmax = 0.9
  val Wmin = 0.4
  var Vmax = 1.0
  var Vmin = -1.0
  var repository = 15 // 非支配解种群数量,存档大小
  var iter_max = 30 // 迭代次数
  val numberOfObjective = 2 // 目标数量
  var numberOfParticles = 50 //粒子数量
  var numberOfFeatures = 19 // numberOfFeatures 即有多少个决策变量（对应数据集中除了代表簇的那一列外还剩有多少列）
  var numberOfClusters = 7 // numClusters 表示一共聚成了多少个簇（即K 值）
  val numberOfKMeansIter = 30 // 设置 kmeans 的运行次数
  var LIndex = 10 // 最近邻的个数设置
  //  val numberOfMerge = 3 // 相隔 numberOfMerge 进行合并
  var numberOfSubPop = 1 // 设置并行度
  val shold = 0.0000000001 //阈值，用于判断聚类中心偏移量


  /*
  * TODO
  *  属于数据并行
  * */
  def main(args: Array[String]): Unit = {
    val dataName = args(0)
    val labelStartWithZero = args(1).toInt
    val inputHDFS: String = "/home/hadoop/linghuidong/single/dataSet/" + dataName
    val outputHDFS: String = "./singleResult/" + dataName + "/"


    val start = System.nanoTime
    val initData: Array[String] = Source.fromFile(inputHDFS).getLines().toArray

    // 处理原始数据
    val allDataWithKArray: Array[(Array[Double], Int)] = handleDataWithOutputDataWithK(initData,
      inputDataWithHeader = false,
      inputDataWithK = true,
      splitString = ",",
      labelStartWithZero)


    // numberOfClusters
    numberOfClusters = allDataWithKArray.map(_._2).distinct.length
    //    numberOfClusters = 5
    val array1 = allDataWithKArray.map(_._1)
    numberOfFeatures = array1(0).length

    // print base info
    val baseSetting = showBaseInfo(allDataWithKArray, numberOfClusters, numberOfFeatures, inputHDFS, outputHDFS)

    // ======== all data in dataset ===============
    val allData = allDataWithKArray.map(_._1)

    val allDataNumbers = allData.length

    // 计算每个特征的上下限
    val featureThreshold: Array[(Double, Double)] = getFeatureThreshold(allData)
    featureThreshold.foreach(println)

    //  初始化粒子群的位置
    // 使用kmeans 算法来初始化种群的位置
    val KMeanCenter: Array[Array[Double]] = initPositionWithKmeans(initData, numberOfClusters, numberOfKMeansIter = numberOfKMeansIter) // 种群位置x
    //    val initPositionArray = initPositionWithRandom(featureThreshold) // 种群位置x
    val array = initPositionWithDataPointMaxDistanceNew(allData, numberOfClusters)
    array.map {
      i =>
        println(i.toList)
    }


    val initPopTemp = ArrayBuffer[(Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])]()
    for (i <- 0 until (numberOfParticles)) {
      val particleOne = Tuple6(
        initPositionWithDataPointMaxDistanceNew(allData, numberOfClusters)
        //        KMeanCenter
        , Array.fill[Double](numberOfClusters, numberOfFeatures)(Random.nextDouble() * (Vmax - Vmin) + Vmin) // 种群速度 v
        , Array.fill[Double](numberOfObjective)(0.0) // 当前的适应度值
        , Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 个体最优位置
        , Array.fill[Double](numberOfObjective)(0.0) // 个体最优值
        , Array.fill[Double](1)(0.0) // 拥挤度
      )
      initPopTemp += particleOne
    }
    var initPopArray = initPopTemp.toArray


    // 初始化个体最优值
    var particleSwarm = initPopArray.map {
      iter =>
        (iter._1, iter._2, iter._3, iter._1, iter._3, iter._6)
    }

    // 计算种群的适应度值
    particleSwarm = particleSwarm.map(i => calFitness(i, allData))
    // 将粒子群中的粒子都复制到存档中，剔除支配解
    var Archive: Array[(Array[Array[Double]], Array[Double], Array[Double])] = particleSwarm.map {
      line =>
        val position: Array[Array[Double]] = line._1
        val fitness: Array[Double] = line._3
        val distance: Array[Double] = line._6
        // 添加到Archive 中
        Tuple3(position, fitness, distance)
    }
    // 剔除支配解
    Archive = updateArchive(Archive)
    val initArchiveHead = Archive.map(_._2).head
    println("初始外部存档解的形状：" + Archive.map(_._2).length + "x" + initArchiveHead.length + ",初始外部存档的适应度值：")
    Archive.map(_._2).map(i => (i(0), i(1))).foreach(println)


    while (iter <= iter_max) {
      // 选取一个全局最优解,选择不那么拥挤的一个作为全局最优解
      val globalBest: (Array[Array[Double]], Array[Double], Array[Double]) = getGlobalBest(Archive)

      // 更新W
      val w = (Wmax - Wmin) * (iter_max - iter) / iter_max + Wmin

      // 更新粒子的速度、位置、适应度值、个体最优
      val newParticleSwarm = particleSwarm.map {
        i =>
          updateParticle(
            i._1,
            i._2,
            i._4,
            i._5,
            globalBest._1,
            i._6,
            allData,
            featureThreshold,
            w)
      }

      val archiveTemp = new ArrayBuffer[(Array[Array[Double]], Array[Double], Array[Double])]()
      Archive.foreach {
        i =>
          archiveTemp += i
      }
      // 将粒子群中的粒子都复制到存档中，剔除支配解
      newParticleSwarm.foreach {
        line =>
          val position: Array[Array[Double]] = line._1
          val fitness: Array[Double] = line._3
          val distance: Array[Double] = line._6
          // 添加到Archive 中
          archiveTemp += Tuple3(position, fitness, distance)
      }
      // 剔除支配解
      Archive = updateArchive(archiveTemp.toArray)

      println("第" + iter + "次迭代后，updateArchive后的大小" + Archive.size)

      iter = iter + 1

    }

    // As it is a nano second we need to divide it by 1000000000. in 1e9d "d" stands for double
    val duration = (System.nanoTime - start) / 1e9d
    println("Timer", duration)


    //    Archive.map {
    //      i =>
    //        println((i._2(0), i._2(1)))
    //    }
    //
    //    // 数据归一化
    //    val archiveNormalization = dataNormalization(Archive)
    //
    //    // 对全部结果进行计算
    //    val allArchiveStringBuffer = ArrayBuffer[String]()
    //    var number = 0
    //    Archive.map(_._1).map {
    //      i =>
    //        allArchiveStringBuffer += "==========new Archive============"
    //        val map = scala.collection.mutable.HashMap.empty[Int, Int]
    //        var calRealNum = 0
    //        for (j <- 1 until (numberOfClusters + 1)) {
    //          val real = allDataWithKArray.filter(_._2 == j)
    //          val tuple = getTrueRateNew(real.map(_._1), i)
    //          calRealNum += tuple._2
    //          map += (tuple._1 -> tuple._2)
    //          allArchiveStringBuffer += "簇：" + tuple._1 + ",占：" + tuple._2 + ",实际：" + tuple._3
    //          println("簇：" + tuple._1 + ",占：" + tuple._2 + ",实际：" + tuple._3)
    //        }
    //        val size = map.size
    //        if (size == numberOfClusters) {
    //          number = number + 1
    //          println("正确的个数：" + calRealNum + ",准确率：" + (calRealNum * 1.0 / allDataNumbers))
    //          allArchiveStringBuffer += "正确的个数：" + calRealNum + ",准确率：" + (calRealNum * 1.0 / allDataNumbers)
    //        }
    //        println
    //        allArchiveStringBuffer += "\n"
    //    }
    //    val allArchiveString: Array[String] = allArchiveStringBuffer.toArray
    //
    //    println
    //    println("kmeans 的解集：")
    //    var calkmeansRealNum = 0
    //    var KmeansAccuracy = 0.0
    //    var kmeansArchiveString: ArrayBuffer[String] = ArrayBuffer[String]()
    //    val kmeansmap = scala.collection.mutable.HashMap.empty[Int, Int]
    //    for (j <- 1 until (numberOfClusters + 1)) {
    //      val real = allDataWithKArray.filter(_._2 == j)
    //      val tuple = getTrueRateNew(real.map(_._1), KMeanCenter)
    //      calkmeansRealNum += tuple._2
    //      kmeansmap += (tuple._1 -> tuple._2)
    //      kmeansArchiveString += "最好的簇：" + tuple._1 + ",占：" + tuple._2 + ",实际：" + tuple._3
    //    }
    //    val kmeanssize = kmeansmap.size
    //    if (kmeanssize == numberOfClusters) {
    //      KmeansAccuracy = (calkmeansRealNum * 1.0 / allDataNumbers)
    //      kmeansArchiveString += "正确的个数：" + calkmeansRealNum + ",准确率：" + KmeansAccuracy
    //
    //    }
    //    kmeansArchiveString += "一共有" + number + "种方案可以选择。"
    //    showAndSaveArchive(Archive,
    //      archiveNormalization,
    //      KMeanCenter,
    //      inputHDFS,
    //      outputHDFS,
    //      duration,
    //      baseSetting,
    //      allArchiveString,
    //      kmeansArchiveString)

    saveParticle(Archive, baseSetting, duration, inputHDFS, outputHDFS)


  }

  def saveParticle(Archive: Array[(Array[Array[Double]], Array[Double], Array[Double])],
                   baseSetting: String,
                   duration: Double,
                   DataSetPath: String,
                   savePath: String): Unit = {
    val outputPath = savePath + "/" + NowDate() + ".txt"
    val file = new File(outputPath)
    file.getParentFile().mkdirs() // 创建目录
    file.createNewFile() // 创建文件


    val outputFile = new PrintWriter(outputPath)
    println(baseSetting)
    println("数据集：" + DataSetPath)
    println("结果存档：" + outputPath)
    outputFile.println("数据集：" + DataSetPath)
    outputFile.println("结果存档：" + outputPath)

    println("运行时间为：" + duration)
    outputFile.println("运行时间为：" + duration)

    println()
    outputFile.println()

    // 对全部结果进行计算

    Archive.map(_._1).map {
      i: Array[Array[Double]] =>
        outputFile.println()
        for (j <- 0 until (numberOfClusters)) {
          outputFile.println(i(j).toList)
        }

    }


    outputFile.close()
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


  def getTrueRateNew(allDataValue: Array[Array[Double]], mopsoCenters: Array[Array[Double]]) = {
    val allDataWithClusterK = allDataValue.map {
      line =>
        val distanceArray: ArrayBuffer[(Int, Double)] = new ArrayBuffer[(Int, Double)]() // 表示：（Int,Distance） ==> 哪个簇，距离
        var k = 1
        for (elem <- mopsoCenters) {
          val distance = dist(line, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK = distanceArray.minBy(_._2)
        (lineK._1, line)
    }
    val value: Array[(Int, Int)] = allDataWithClusterK.map {
      i =>
        val k = i._1
        (k, 1)
    }

    val result: Map[Int, Int] = value.groupBy(_._1).map {
      i =>
        val length = i._2.length
        (i._1, length)
    }
    val allDataSum = result.map(_._2).sum
    val maxCount: (Int, Int) = result.maxBy(_._2)
    Tuple3(maxCount._1, maxCount._2, allDataSum)
    //    value.foreach(println)
    //    value.repartition(1).saveAsTextFile(outputHDFS + "\\TrueRate\\" + NowDate())
  }


  // ============== 根据轮廓系数从Archive 中选择一个最佳的解作为最后的解==============
  def selectBestArchiveAsFinalResult(Archive: Array[(Array[Array[Double]], Array[Double], Array[Double])],
                                     inputData: Array[(Array[Double], Int)]) = {
    val ArchiveCenters: Array[Array[Array[Double]]] = Archive.map(_._1)
    val silhouetteCoefficientArray = ArrayBuffer[Double]()
    val allCenterSilhouetteCoefficientSum: Array[(Array[Array[Double]], Double)] = ArchiveCenters.map {
      centers: Array[Array[Double]] =>

        val silhouetteCoefficientSumMean = silhouetteCoefficient(inputData, centers)

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
    //    DescendingOrder
    (silhouetteCoefficientArray.toArray, AscendingOrder)
  }

  // ================= 轮廓系数均值 ============
  def silhouetteCoefficient(inputData: Array[(Array[Double], Int)], center: Array[Array[Double]]) = {

    val inputDataWithPoints: Array[Array[Double]] = inputData.map(_._1)

    val allPointWithCalK = calKmeans(inputDataWithPoints, center)

    val allPointWithKArray: Array[(Int, Array[Double])] = allPointWithCalK.map(i => (i._1._1, i._2))
    val length = allPointWithKArray.length

    var S = 0.0
    for (i <- 0 until length) {
      val a = allPointWithKArray(i)
      val aArray = allPointWithKArray.filter(_._1 == a._1)
      var asum = 0.0
      var alength = 0
      aArray.foreach {
        i =>
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

  // =====================更新粒子的速度、位置、适应度值、个体最优===================
  def updateParticle(position: Array[Array[Double]],
                     velocity: Array[Array[Double]],
                     BestPosition: Array[Array[Double]],
                     BestFitness: Array[Double],
                     globalBestPosition: Array[Array[Double]],
                     CrowdingDistance: Array[Double],
                     allData: Array[Array[Double]],
                     featureThreshold: Array[(Double, Double)],
                     w: Double) = {

    val newPosition: Array[Array[Double]] = Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 更新后的位置
    val newVelocity: Array[Array[Double]] = Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 更新后的速度
    var newBestPosition: Array[Array[Double]] = Array.fill[Double](numberOfClusters, numberOfFeatures)(0.0) // 更新后的个体最优位置
    var newBestFitness = Array.fill(numberOfObjective)(0.0) // 更新后个体最优适应度值

    // 粒子速度和位置更新
    List.range(0, numberOfClusters).foreach {
      i =>
        List.range(0, numberOfFeatures).foreach {
          j =>
            // 更新速度信息
            newVelocity(i)(j) = w * velocity(i)(j) + c1 * Random.nextDouble() * (BestPosition(i)(j) - position(i)(j)) + c2 * Random.nextDouble() * (globalBestPosition(i)(j) - position(i)(j))

            if (newVelocity(i)(j) > Vmax || newVelocity(i)(j) < Vmin) {
              newVelocity(i)(j) = Random.nextDouble() * (Vmax - Vmin) + Vmin
            }

            // 更新位置信息
            newPosition(i)(j) = position(i)(j) + newVelocity(i)(j)

            if (newPosition(i)(j) > featureThreshold(j)._1 || newPosition(i)(j) < featureThreshold(j)._2) {
              newPosition(i)(j) = Random.nextDouble() * (featureThreshold(j)._1 - featureThreshold(j)._2) + featureThreshold(j)._2
            }
        }
    }

    // 计算出更新后粒子的适应度值
    val newAllDataWithClusterK: Array[((Int, Double), Array[Double])] = calKmeans(allData, newPosition)
    val newDevValue = Dev(newAllDataWithClusterK)
    val newConnValue = Conn(newAllDataWithClusterK)
    val newFitness = Array(newDevValue, newConnValue) // 更新后的适应度值

    // 如果当前解优于（支配）当前局部最优副本则替换之；
    newBestFitness = BestFitness
    newBestPosition = BestPosition

    if (isDominatedBy(newFitness, BestFitness)) {
      // 两者互不支配，则随机决定是否用当前解替换局部最优解
      newBestFitness = newFitness
      newBestPosition = newPosition
    } else {
      if (Random.nextDouble() < 0.5) {
        newBestFitness = newFitness
        newBestPosition = newPosition
      }
    }

    // 突变


    (newPosition, newVelocity, newFitness, newBestPosition, newBestFitness, CrowdingDistance)
  }

  // =========== 计算适应度值 ==============
  def calFitness(particles: (Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])
                 , allData: Array[Array[Double]]) = {
    val allDataWithClusterK: Array[((Int, Double), Array[Double])] = calKmeans(allData, particles._1)
    val DevValue = Dev(allDataWithClusterK)
    val ConnValue = Conn(allDataWithClusterK)
    val fitness = Array(DevValue, ConnValue)
    (particles._1, particles._2, fitness, particles._1, fitness, particles._6)
  }


  // =============== 根据欧几里得距离将数据集中的点指定给最近的质心 ===================================
  def calKmeans(allDataValue: Array[Array[Double]], clusterCenter: Array[Array[Double]]) = {
    allDataValue.map {
      line =>
        val distanceArray: ArrayBuffer[(Int, Double)] = new ArrayBuffer[(Int, Double)]() // 表示：（Int,Distance） ==> 哪个簇，距离
        var k = 1
        for (elem <- clusterCenter) {
          val distance = dist(line, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK = distanceArray.minBy(_._2)
        (lineK, line)
    }
  }


  // ====================== 初始化粒子群，适应kmean 作为初始的粒子群 ==========================
  def initPositionWithKmeans(input: Array[String], k: Int, numberOfKMeansIter: Int) = {
    val result: Array[Array[Double]] = Array.ofDim[Double](numberOfClusters, numberOfFeatures)
    val points = input.map(line => { //数据预处理
      val parts = line.split(",").map(_.toDouble)
      var vector = Vector[Double]()
      for (i <- 0 until (numberOfFeatures))
        vector ++= Vector(parts(i))
      vector
    })
    val singleKmeans = new SingleKmeans(k, numberOfFeatures)
    val initCenter = singleKmeans.initialCenters(points)
    val center = singleKmeans.kmeans(points, initCenter)
    var index = 0
    center.foreach {
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
  }


  // ====================== 初始化粒子群，随机从数据中选取 ==========================
  def initPositionWithDataPoint(input: Array[Array[Double]], k: Int) = {
    val result = ArrayBuffer[Array[Double]]()
    Array.range(0, k).foreach {
      i =>
        result += input(Random.nextInt(input.length))
    }
    result.toArray
  }

  // =============== 初始化粒子的位置 ,返回的大小为：numberOfClusters * numberOfFeatures =============
  def initPositionWithRandom(featureThreshold: Array[(Double, Double)]) = {
    val result: Array[Array[Double]] = Array.ofDim[Double](numberOfClusters, numberOfFeatures)
    for (i <- 0 until (numberOfClusters)) {
      for (j <- 0 until (numberOfFeatures)) {
        result(i)(j) = Random.nextDouble() * (featureThreshold(j)._1 - featureThreshold(j)._2) + featureThreshold(j)._2
      }
    }
    result
  }

  // ====================== 初始化粒子群，随机从数据中选取 ==========================
  def initPositionWithDataPointMaxDistance(input: Array[Array[Double]], numberOfCluster: Int) = {

    val result = ArrayBuffer[Array[Double]]()
    val length = input.length
    val firstCenter: Array[Double] = input(Random.nextInt(length))
    result += firstCenter

    for (i <- 1 until numberOfCluster) {

      val tuples: Array[((Int, Double), Array[Double])] = calKmeans(input, result.toArray)
      val tuples1: Array[(Double, Array[Double])] = tuples.map(i => (i._1._2, i._2))
      val otherCenter = tuples1.maxBy(_._1)._2
      //      val otherCenter = tuples.map(i => (i._1._2, i._2)).maxBy(_._1)._2
      result += otherCenter

    }
    // 升序
    //    val AscendingOrder = result.toArray.sortBy(i => i(0))
    // 降序
    //    val DescendingOrder = AscendingOrder.reverse
    //    DescendingOrder
    //    AscendingOrder
    result.toArray
  }

  // ====================== 初始化粒子群，随机从数据中选取 ==========================
  def initPositionWithDataPointMaxDistanceNew(input: Array[Array[Double]], numberOfCluster: Int) = {

    val result = ArrayBuffer[Array[Double]]()
    val length = input.length
    val firstCenter: Array[Double] = input(Random.nextInt(length))
    result += firstCenter

    // 先求第二个聚类中心
    val firstDistance = input.map {
      i =>
        val d = dist(i, firstCenter)
        (i, d)
    }
    val secondCenter = firstDistance.maxBy(_._2)._1
    result += secondCenter

    // 三个点及以后，
    // 然后再选择距离前两个点的最短距离最大的那个点作为第三个初始类簇的中心点
    for (i <- 2 until numberOfCluster) {
      val otherCenter: Array[Double] = input.map {
        i =>
          val jToCenter = result.map {
            centerI =>
              val d = dist(centerI, i)
              (d, centerI)
          }
          val tuple: (Double, Array[Double]) = jToCenter.minBy(_._1)
          (tuple._1, i)
      }.maxBy(_._1)._2
      result += otherCenter
    }


    result.toArray
  }


  // ===================== 更新粒子群的位置、速度 =======================================
  def updateParticleSwarmVelocityAndPosition(particleSwarmFitness: Array[(Array[Array[Double]], Array[Array[Double]], Array[Double], Array[Array[Double]], Array[Double], Array[Double])],
                                             globalBest: (Array[Array[Double]], Array[Double], Array[Double]),
                                             featureThreshold: Array[(Double, Double)],
                                             w: Double) = {
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

                if (newPosition(i)(j) > featureThreshold(j)._1 || newPosition(i)(j) < featureThreshold(j)._2) {
                  newPosition(i)(j) = Random.nextDouble() * (featureThreshold(j)._1 - featureThreshold(j)._2) + featureThreshold(j)._2
                }
            }
        }
        (newPosition, newVelocity, particle._3, particle._4, particle._5, particle._6)
    }
    updateParticleSwarmWithVelocityAndPosition
  }


  // =============== 根据欧几里得距离将数据集中的点指定给最近的质心 ===================================
  def calTrueKWithCalKKmeans(allDataValue: Iterator[(Array[Double], Int)], clusterCenter: Array[Array[Double]]) = {
    allDataValue.map {
      line =>
        // 表示：（Int,Distance） ==> 哪个簇，距离
        val distanceArray = new ArrayBuffer[(Int, Double)]()
        var k = 1
        for (elem <- clusterCenter) {
          val distance = dist(line._1, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK: (Int, Double) = distanceArray.minBy(_._2)
        (line._2, lineK._1)
    }
  }


  // ======== 输出到控制台并保存 ========================
  def showAndSaveArchive(Archive: Array[(Array[Array[Double]], Array[Double], Array[Double])],
                         archiveNormalization: Array[(Array[Array[Double]], Array[Double], Array[Double])],
                         KmeansPosition: Array[Array[Double]],
                         DataSetPath: String,
                         savePath: String,
                         duration: Double,
                         baseSetting: String,
                         allArchiveString: Array[String],
                         kmeansArchiveString: ArrayBuffer[String]): Unit = {
    val outputPath = savePath + "/" + NowDate() + ".txt"
    val file = new File(outputPath)
    file.getParentFile().mkdirs() // 创建目录
    file.createNewFile() // 创建文件


    val outputFile = new PrintWriter(outputPath)
    println(baseSetting)
    println("数据集：" + DataSetPath)
    println("结果存档：" + outputPath)
    outputFile.println("数据集：" + DataSetPath)
    outputFile.println("结果存档：" + outputPath)

    println("运行时间为：" + duration)


    println()
    outputFile.println()
    val head = Archive.map(_._2).head
    println("外部存档解的形状：" + Archive.map(_._2).length + "x" + head.length + ",外部存档的适应度值：")
    outputFile.println("外部存档解的形状：" + Archive.map(_._2).length + "x" + head.length + ",外部存档的适应度值：")
    Archive.map(_._2).map(i => (i(0), i(1))).foreach(println)
    // 保存适应度值
    Archive.map(i => (i._2(0), i._2(1))).foreach {
      i =>
        val str = i._1 + "," + i._2
        outputFile.println(str)
    }


    println()
    outputFile.println()
    val archiveNormalizationHead = archiveNormalization.map(_._2).head
    outputFile.println("归一化后外部存档解的形状：" + archiveNormalization.map(_._2).length + "x" + archiveNormalizationHead.length + ",外部存档的适应度值：")
    println("归一化后外部存档解的形状：" + archiveNormalization.map(_._2).length + "x" + archiveNormalizationHead.length + ",外部存档的适应度值：")
    archiveNormalization.map(_._2).map(i => (i(0), i(1))).foreach(println)
    // 保存适应度值
    archiveNormalization.map(i => (i._2(0), i._2(1))).foreach {
      i =>
        val str = i._1 + "," + i._2
        outputFile.println(str)
    }

    // 输出簇中心的位置
    println()
    outputFile.println()
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
        outputFile.println("MOPSO Center Point of Cluster " + (clusterIndex + 1) + "==》  " + s)
        clusterIndex += 1
      }
      outputFile.println()
      println()
    }

    outputFile.println()
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
        outputFile.println("Kmeans Center Point of Cluster " + (clusterIndex + 1) + "==》  " + lineString)
        clusterIndex += 1
    }

    outputFile.println()
    println()
    outputFile.println("所有存档的分布情况：")
    println("所有存档的分布情况：")
    allArchiveString.foreach {
      i =>
        outputFile.println(i)
    }
    outputFile.println("\n")

    outputFile.println("\nkmeans 的分布情况")
    kmeansArchiveString.foreach {
      i =>
        outputFile.println(i)
    }


    outputFile.println("\n外部存档解的形状：" + Archive.map(_._2).length)


    outputFile.close()
  }


  // ========================= 获取当前时间 =====================
  def NowDate(): String = {
    val now: Date = new Date()
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss")
    val date = dateFormat.format(now)
    date
  }

  // =============== 从所有分区的适应度中选择一个合适的适应度 ================
  def selectPartitionBestFitness(fitnesss: Array[Array[Double]]): Array[Double] = {
    val partitionCrowdingDistance = calculatePartitionCrowdingDistance(fitnesss)
    val size = partitionCrowdingDistance.size
    if (size == 1) {
      val result = partitionCrowdingDistance(0)._1
      return result
    }
    if (size == 2) {
      val i = Math.abs(new Random().nextInt(2))
      val result = partitionCrowdingDistance(i)._1
      return result
    }
    // 计算最大拥挤距离
    val archiveCrowdingDistanceRemoveFirst = partitionCrowdingDistance.drop(1) // 避开值为无穷大的两个端点）
    val archiveCrowdingDistanceRemoveFirstAndRemoveLast = archiveCrowdingDistanceRemoveFirst.dropRight(1) // 避开值为无穷大的两个端点）
    val MaxCrowdingDistance: Double = archiveCrowdingDistanceRemoveFirstAndRemoveLast.maxBy(_._2(0))._2(0)

    // 获取拥挤距离最大的子集（避开值为无穷大的两个端点）
    // 找到拥挤度最大的子集
    val leastCrowdedArrayBuffer = ArrayBuffer[(Array[Double], Array[Double])]()
    partitionCrowdingDistance.foreach {
      i =>
        if (i._2(0) == MaxCrowdingDistance) {
          leastCrowdedArrayBuffer += i
        }
    }
    val leastCrowded = leastCrowdedArrayBuffer.toArray

    // 从子集中随机返回一个粒子
    val index = Random.nextInt(leastCrowded.size)
    val result = leastCrowded(index)._1
    result
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
  def calFitnessNew(allData: Array[Array[Double]], positions: Array[Array[Double]]) = {

    val allDataWithClusterK = allData.map {
      line =>
        val distanceArray: ArrayBuffer[(Int, Double)] = new ArrayBuffer[(Int, Double)]() // 表示：（Int,Distance） ==> 哪个簇，距离
        var k = 1
        for (elem <- positions) {
          val distance = dist(line, elem)
          distanceArray += Tuple2(k, distance)
          k = k + 1
        }
        // 找到最小的距离
        val lineK = distanceArray.minBy(_._2)
        (lineK, line)
    }

    val DevValue = Dev(allDataWithClusterK)
    val ConnValue = Conn(allDataWithClusterK)
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
    var updateArchiveIng = updateArchiveIngArrayBuffer.toArray

    val overflow = updateArchiveIng.size - repository
    // 如果存档集合溢出则截断之（剔除最拥挤的粒子）

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
      val d = Math.abs(pre._2(0) - next._2(0)) * Math.abs(pre._2(1) - next._2(1))
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
  }


  // ===================== 处理数据，输出数据带k========================================
  def handleDataWithOutputDataWithK(input: Array[String],
                                    inputDataWithHeader: Boolean,
                                    inputDataWithK: Boolean,
                                    splitString: String,
                                    labelStartWithZeroBCValue: Int): Array[(Array[Double], Int)] = {
    var inputData = input

    // 如果你的csv文件有标题 的话，需要剔除首行
    if (inputDataWithHeader) { // 数据文件有标题 的话，需要剔除首行
      val header = input(0)
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


  //  =============================== 显示基础信息 =============================
  //     showBaseInfo(inputData,numberOfClusters,numberOfFeatures,inputHDFS,outputHDFS)
  def showBaseInfo(inputData: Array[(Array[Double], Int)],
                   numberOfClustersInput: Int,
                   numberOfFeaturesInput: Int,
                   inputHDFSInput: String,
                   outputHDFSInput: String) = {
    // 基础参数设置
    val baseSetting = "模型：" + this.getClass.getName + ",\n粒子的个数：" + numberOfParticles + ", 存档大小:" + repository + ", k 值：" + numberOfClustersInput + ",特征数：" +
      numberOfFeaturesInput + ",最近邻: " + LIndex + ", 迭代次数：" +
      iter_max + ", 并行度：" + numberOfSubPop
    println(baseSetting)
    baseSetting
  }

  // ================== 得到每个特征的上下限 ================================
  def getFeatureThreshold(inputDataValueRDD: Array[Array[Double]]) = {
    val tuples = ArrayBuffer[(Double, Double)]()

    for (i <- 0 until (numberOfFeatures)) {
      val featureI: Array[Double] = inputDataValueRDD.map {
        j =>
          j(i)
      }
      val min = featureI.min
      val max: Double = featureI.max
      tuples += Tuple2(max, min)
    }
    tuples.toArray
  }

  // ================= Davies-Bouldin Index(戴维森堡丁指数)(分类适确性指标)(DB) ============
  def DBI(inputData: Array[(Array[Double], Int)], clusterCenter: Array[Array[Double]]) = {
    val length = clusterCenter.length
    val inputDataWithPoints: Array[Array[Double]] = inputData.map(_._1)

    val dataWithCalKAndDistance: Array[((Int, Double), Array[Double])] = calKmeans(inputDataWithPoints, clusterCenter)

    val dataWithCalK: Array[(Int, Double)] = dataWithCalKAndDistance.map(_._1)
    var max = Double.MinValue
    var result = 0.0

    val intToTuples: Map[Int, Array[(Int, Double)]] = dataWithCalK.groupBy(_._1)

    val withinClusterDistance: Array[(Int, Double)] = intToTuples.map {
      i =>
        val value: Array[(Int, Double)] = i._2
        val length1 = value.length
        val sum = value.map(_._2).sum
        (i._1, sum / length1)
    }.toArray

    for (i <- 0 until (length)) {
      val si = withinClusterDistance.filter(_._1 == (i + 1))(0)
      for (j <- i until (length)) {
        if (i != j) {
          val sj = withinClusterDistance.filter(_._1 == (j + 1))(0)
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
  def calInertia(inputData: Array[(Array[Double], Int)], center: Array[Array[Double]]) = {
    val inputDataWithPoints: Array[Array[Double]] = inputData.map(_._1)

    val allDataWithCalK: Array[((Int, Double), Array[Double])] = calKmeans(inputDataWithPoints, center)

    val result = allDataWithCalK.map {
      i => (i._1._2 * i._1._2)
    }.sum

    result
  }


  class SingleKmeans(input_k: Int, input_dim: Int) {
    val k = input_k //类个数
    val dim = input_dim //数据集维度
    val shold = 0.0000000001 //阈值，用于判断聚类中心偏移量

    //-------------------------------------随机初始化聚类中心---------------------------------------------------
    def initialCenters(points: Array[Vector[Double]]) = {
      val centers = new Array[Vector[Double]](k) //聚类中心点（迭代更新）
      val pointsNum = points.length //数据集个数
      //寻找k个随机数(作为数据集的下标)
      val random = new Random()
      var index = 0
      var flag = true
      var temp = 0
      var array = new mutable.LinkedList[Int]() //保存随机下标号
      while (index < k) {
        temp = new Random().nextInt(pointsNum)
        flag = true
        if (array.contains(temp)) { //在数组中存在
          flag = false
        }
        else {
          if (flag) {
            array = array :+ temp
            index += 1
          }
        } //else-end
      } //while-end
      for (i <- 0 to centers.length - 1) {
        centers(i) = points(array(i))
      }

      centers
    }

    //---------------------------迭代做聚类-------------------------------------
    def kmeans(points: Array[Vector[Double]], centers: Array[Vector[Double]]) = {
      var bool = true
      var newCenters = Array[Vector[Double]]()
      var move = 0.0
      var currentCost = 0.0 //当前的代价函数值
      var newCost = 0.0
      //根据每个样本点最近的聚类中心进行groupBy分组，最后得到的cluster是Map[Vector[Double],Array[Vector[Double]]]
      //Map中的key就是聚类中心，value就是依赖于该聚类中心的点集
      while (bool) { //迭代更新聚类中心，直到最优
        move = 0.0
        currentCost = computeCost(points, centers)
        val cluster = points.groupBy(v => closestCenter(centers, v))
        newCenters =
          centers.map(oldCenter => {
            cluster.get(oldCenter) match { //找到该聚类中心所拥有的点集
              case Some(pointsInThisCluster) =>
                //均值作为新的聚类中心
                vectorDivide(pointsInThisCluster.reduceLeft((v1, v2) => vectorAdd(v1, v2)), pointsInThisCluster.length)
              case None => oldCenter
            }
          })
        for (i <- 0 to centers.length - 1) {
          //move += math.sqrt(vectorDis(newCenters(i),centers(i)))
          centers(i) = newCenters(i)
        }

        newCost = computeCost(points, centers) //新的代价函数值

        if (math.sqrt(vectorDis(Vector(currentCost), Vector(newCost))) < shold)
          bool = false
      } //while-end

      centers
    }

    //--------------------------输出聚类结果-----------------------------
    def printResult(points: Array[Vector[Double]], centers: Array[Vector[Double]]) = {
      //将每个点的聚类中心用centers中的下标表示，属于同一类的点拥有相同的下标
      val pointsNum = points.length
      val pointsLabel = new Array[Int](pointsNum)
      var closetCenter = Vector[Double]()
      println("聚类结果如下：")
      for (i <- 0 to pointsNum - 1) {
        closetCenter = centers.reduceLeft((c1, c2) => if (vectorDis(c1, points(i)) < vectorDis(c2, points(i))) c1 else c2)
        pointsLabel(i) = centers.indexOf(closetCenter)
        println(points(i) + "-----------" + pointsLabel(i))
      }

    }

    //--------------------------找到某样本点所属的聚类中心-----------------------------
    def closestCenter(centers: Array[Vector[Double]], v: Vector[Double]): Vector[Double] = {
      centers.reduceLeft((c1, c2) =>
        if (vectorDis(c1, v) < vectorDis(c2, v)) c1 else c2
      )
    }

    //--------------------------计算代价函数（每个样本点到聚类中心的距离之和不再有很大变化）-----------------------------
    def computeCost(points: Array[Vector[Double]], centers: Array[Vector[Double]]): Double = {
      //cluster:Map[Vector[Double],Array[Vector[Double]]
      val cluster = points.groupBy(v => closestCenter(centers, v))
      var costSum = 0.0
      //var subSets = Array[Vector[Double]]()
      for (i <- 0 to centers.length - 1) {
        cluster.get(centers(i)) match {
          case Some(subSets) =>
            for (j <- 0 to subSets.length - 1) {
              costSum += (vectorDis(centers(i), subSets(j)) * vectorDis(centers(i), subSets(j)))
            }
          case None => costSum = costSum
        }
      }
      costSum
    }

    //--------------------------自定义向量间的运算-----------------------------
    //--------------------------向量间的欧式距离-----------------------------
    def vectorDis(v1: Vector[Double], v2: Vector[Double]): Double = {
      var distance = 0.0
      for (i <- 0 to v1.length - 1) {
        distance += (v1(i) - v2(i)) * (v1(i) - v2(i))
      }
      distance = math.sqrt(distance)
      distance
    }

    //--------------------------向量加法-----------------------------
    def vectorAdd(v1: Vector[Double], v2: Vector[Double]): Vector[Double] = {
      var v3 = v1
      for (i <- 0 to v1.length - 1) {
        v3 = v3.updated(i, v1(i) + v2(i))
      }
      v3
    }

    //--------------------------向量除法-----------------------------
    def vectorDivide(v: Vector[Double], num: Int): Vector[Double] = {
      var r = v
      for (i <- 0 to v.length - 1) {
        r = r.updated(i, r(i) / num)
      }
      r
    }
  }

}

