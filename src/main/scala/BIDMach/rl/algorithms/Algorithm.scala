package BIDMach.rl.algorithms;

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks.layers._;
import BIDMach.networks._
import BIDMach.updaters._
import BIDMach._
import BIDMach.rl.environments._
import BIDMach.rl.estimators._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;


@SerialVersionUID(100L)
abstract class Algorithm(opts:Algorithm.Opts = new Algorithm.Options) extends Serializable {
  
	var myLogger = Mat.consoleLogger;
	var fut:Future[_] = null;
	
  def startup;
  
  def train;
  
  def launchTrain = {
    val tmp = myLogger;
    myLogger = Mat.getFileLogger(opts.logfile);
  	val executor = Executors.newFixedThreadPool(opts.nthreads);
  	val runner = new Runnable{
  	  def run() = {
  	    try {
  	    	train;
  	    } catch {
  	      case e:Throwable => myLogger.severe("Training thread failed: %s" format Learner.printStackTrace(e));
  	    }
    	  myLogger = tmp;
    	}
  	}
  	fut = executor.submit(runner);
  	fut;
  }
}

object Algorithm {
  trait Opts extends Net.Opts with ADAGrad.Opts { 	
  	clipByValue = 1f;                                // gradient clipping
  	gsq_decay = 0.99f;                               // Decay factor for MSProp
  	vel_decay = 0.0f;                                // Momentum decay
  	texp = 0f;
  	vexp = 1f;
  	waitsteps = -1;
  	var logfile = "log.txt";
  	var nthreads = 4;
  }
  
  class Options extends Opts {}   
}
