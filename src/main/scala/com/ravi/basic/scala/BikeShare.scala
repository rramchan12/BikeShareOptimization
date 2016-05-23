package com.ravi.basic.scala

import org.apache.spark.{SparkContext,SparkConf}
import org.apache.spark.sql.SQLContext
//Import for Data Frames
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
//Import for ML
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import scala.util.grammar.LabelledRHS
// Imports for Naive Bayes
import org.apache.spark.mllib.classification.NaiveBayes
// Imports for Decision Treees 
import org.apache.spark.mllib.tree.DecisionTree
// Imports for Random Forest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.RandomForest



object BikeShare {
  
   //println(convertDur("14h 26min. 2sec."))    
    case class Trip (id: String, dur: Long, s0:String, s1: String, reg:String)
    case class TripReport ( predicted:Int, observed:Int, model:String, frequency:Double)
    
   val label_map = Map("Registered" -> 0.0 , "Casual" -> 1.0)
  
   
   def main ( args : Array[String]){
    
    System.setProperty("hadoop.home.dir","C:\\ITBox\\hadoop")
    val config = new SparkConf().setAppName("BikeShareDataMap").setMaster("local")
    
    val sc = new SparkContext(config);
   // val sc = new SQLContext(baseContext)
    
    println(sc.getClass().getPackage().getImplementationVersion())
    
    val raw_trips = sc.textFile("src/main/resources/2010-Q4-cabi-trip-history-data.csv")
    raw_trips.take(4).foreach { x => println(x) }
    
    def convertDur(dur : String) : Long ={
      val dur_regex = """(\d+)h\s(\d+)min.\s(\d+)sec.""".r
      val dur_regex(hour, min, second) = dur
      return (hour.toLong*3600L) + (min.toLong*60L) + (second.toLong) 
      
    }
    
   
    
    val bike_trips = raw_trips.map(_.split(","))
   // bike_trips.collect().foreach { x => x.foreach { line => print(line) } }
    
    
    //Remove the header row 
    
    val bike_trips_1 = bike_trips.filter { x => x(0) != "Duration" }
    
    // Map to rdd 
    
    val bike_trips_2 = bike_trips_1.map(r => Trip(r(5), convertDur(r(0)), r(3), r(4), r(6)))
        
    //Caching so that it can be reused
        
    bike_trips_2.cache()
    
    //printing
    
    bike_trips_2.take(5).foreach { x => println(x.s0)
      println(x.s1)}
    
    //Creating Data Frames 
    
    
    val scc = new SQLContext(sc)
    import scc.implicits._
    
    /*val bike_trips_df = bike_trips_2.toDF()
    bike_trips_df.registerTempTable("bikeshare") 
    scc.sql("SELECT * FROM bikeshare LIMIT 10").show()    
    scc.sql("SELECT COUNT(*) AS num, s0, s1 FROM bikeshare GROUP BY s0, s1 ORDER BY num DESC LIMIT 10").show()
    bike_trips_df.explain()
    
    // The average duration is very high for teh causal class 
    bike_trips_df.groupBy("reg").agg(bike_trips_df("reg"), avg(bike_trips_df("dur"))).show()*/
  
    //Create a Training Set 
  //We can create a classifier with Member being the one being predicted and the duration being the Independant variable
  //Start and End Station are parts of Feature Vectors 
  
  val station_map = bike_trips_2.map(_.s0).union(bike_trips_2.map(_.s1)).distinct().zipWithUniqueId().collectAsMap()
  
 station_map.take(5).foreach{f => println(f._1)  
      println(f._2)}
    
    
  
   
    
   var l_bike = bike_trips_2.map 
   { 
      t =>
     var s0 = station_map.get(t.s0).getOrElse(0L).toDouble
     var s1 = station_map.get(t.s1).getOrElse(0L).toDouble
      LabeledPoint(label_map(t.reg), Vectors.dense(t.dur, s0, s1))
    
   }
   
   
   //(casual/reg [duration, (id of station1), (id of station2)]) = the Key is call Label and the Vector is called Features
  l_bike.take(3).foreach { x => println(x) }
  
     //Split this into Train Set and a Test Set in the ratio of 60 - 40
  val splits = l_bike.randomSplit(Array(0.6,0.4), seed  = 11L )
  
  val train_set = splits(0).cache()
  val test_set = splits(1).cache()
  
  val n = test_set.count();
  
  
  //Predicitivity using Naive Bayes 
  //Create the model 
  val model = NaiveBayes.train(train_set, lambda = 1.0)
  //We now tell Naive Bayes to predict and store it against the actual value
  // (prediction, actual)
  val pred = test_set.map( t => (model.predict(t.features),t.label))
  pred.take(5).foreach{println}
  //Count by value = Return the count of each unique value in this RDD as a map of (value, count) pairs.
  
  
  //ConfusionMatrix = ([Predicted Value, Actual Value ], Count 
  
  val cm = sc.parallelize(pred.countByValue().toSeq)
  cm.take(5).foreach{println}
  
  //Peel out what was ((predicted, actual) , 1.0 *Count/total) order by (predicted,actual)
  val cm_nb = cm.map(x => (x._1, (1.0*x._2/n))).sortBy(_._1,true)
  //Around (0.0 - casual = 68%)
  println("Naive Bayes")
  cm_nb.foreach(println)
  
  //Predictivity using Decision Trees
  val numClasses = 2
  val impurity = "gini"
  val maxDepth = 5
  val maxBins = 32
  val categoricalFeaturesInfo = Map[Int, Int]()
  
  val modelDecTree = DecisionTree.trainClassifier(train_set, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins) 
   val predDecTree = test_set.map(t=> (modelDecTree.predict(t.features), t.label))
   val cmDecTree = sc.parallelize(predDecTree.countByValue().toSeq)
   val cm_dt = cmDecTree.map(x => (x._1, (1.0*x._2/n))).sortBy(_._1,true)
   //Accuracy of Casual (0.0, 0.0) is 77%
   println("Decision Tree")
   cm_dt.foreach(println)
   
  //Predictivity Using Random Forests
   val numTrees = 2 //Use more in practice
   val featureSubsetStrategy = "auto"
   
   
   val modelRandomForest = RandomForest.trainClassifier(train_set, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val predRandomForest = test_set.map(t=> (modelDecTree.predict(t.features), t.label))
   val cmRandomForest = sc.parallelize(predDecTree.countByValue().toSeq)
   val cm_rf = cmRandomForest.map(x => (x._1, (1.0*x._2/n))).sortBy(_._1,true)
   //Accuracy of Casual (0.0, 0.0) is 77%
   println("Random Forest")
   cm_rf.foreach(println)
   
   
   //Compare the results 
   //Refer to Trip Report Class above
   
   
   
   val part0 = cm_nb.map(x =>TripReport(x._1._1.toInt, x._1._2.toInt, "0.NB" , x._2))
     val part1 = cm_dt.map(x =>TripReport(x._1._1.toInt, x._1._2.toInt, "1.DT" , x._2))
     val part2 = cm_rf.map(x =>TripReport(x._1._1.toInt, x._1._2.toInt, "1.RF" , x._2))
     
     //Unified Report in Data Frame
     val cm_df = part0.union(part1).union(part2).toDF()
     cm_df.sort("predicted","observed","model").show()
    }
}