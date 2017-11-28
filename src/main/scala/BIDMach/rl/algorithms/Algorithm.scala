package BIDMach.rl.algorithms;

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,Image,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
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
  var done = false;
  var animation_done = true;
  var paused = false;
  var pauseAt = -1L;
  var istep = 0;
  
  var save_length = 0;
	var saved_frames:FMat = null;
	var saved_actions:IMat = null;
	var saved_preds:FMat = null; 
	var reward_plot:FMat = null;
	
  def startup;
  
  def train;
  
  def pause = {
    paused = true;
    Thread.sleep(500);
  }
  
  def unpause = {
    paused = false;
  }
  
  def stop = {
    done = true;
  }
  
  def animate(rate:Float = 30f, iscale:Int=255) = {
    val h = saved_frames.dims(0);
    val w = saved_frames.dims(1);
    val img = Image(saved_frames(?,?,0).reshapeView(h, w)*iscale);
    img.show;
    animation_done = false;
    val runme = new Runnable {
      def run() = {
        var i = 0;
        while (!animation_done) {
          if (i >= saved_frames.dims(2)) i = 0;
        	img.redraw(saved_frames(?,?,i).reshapeView(h, w)*iscale);
        	Thread.sleep((1000f/rate).toInt);
        	i += 1;
        }        
      }
    }
    fut = Image.getService.submit(runme);
  }
  
  def stop_animation = {
    animation_done = true;
  }
  
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
  	gsq_decay = 0.99f;                               // Decay factor for MSProp
  	vel_decay = 0.0f;                                // Momentum decay
  	texp = 0f;
  	vexp = 1f;
  	waitsteps = -1;
  	var logfile = "log.txt";
  	var nthreads = 4;
  	var save_length = 10000;
  }
  
  class Options extends Opts {}   
}
