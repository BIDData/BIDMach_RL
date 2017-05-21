package BIDMach.rl.estimators

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.datasources._
import BIDMach.updaters._
import BIDMach.mixins._
import BIDMach.models._
import BIDMach._
import BIDMach.rl.algorithms._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;
import BIDMach.networks._

@SerialVersionUID(100L)
abstract class Estimator(opts:Algorithm.Options = new Algorithm.Options) extends Serializable {
    
    def createNet:Net;
    
    def formatStates(states:FMat):FMat = {states}
    
    val net:Net = null;
    
    val adagrad = new ADAGrad(opts);
    
/** Perform the initialization that is normally done by the Learner */

    var initialized = false;
    
    def checkinit(states:FMat, actions:IMat, rewards:FMat) = {
    	if (net.mats.asInstanceOf[AnyRef] == null) {
    		net.mats = new Array[Mat](3);
    		net.gmats = new Array[Mat](3);
    	}
    	net.mats(0) = states;
    	if (net.mats(1).asInstanceOf[AnyRef] == null) {
    		net.mats(1) = izeros(1, states.ncols);              // Dummy action vector
    		net.mats(2) = zeros(1, states.ncols);               // Dummy reward vector
    	}
    	if (actions.asInstanceOf[AnyRef] != null) {
    		net.mats(1) <-- actions;
    	}
    	if (rewards.asInstanceOf[AnyRef] != null) {
    		net.mats(2) <-- rewards;
    	}
    	if (!initialized) {
    		net.useGPU = (opts.useGPU && Mat.hasCUDA > 0);
    		net.init();
    		adagrad.init(net);
    		initialized = true;
    	}
    	net.copyMats(net.mats, net.gmats);
    	net.assignInputs(net.gmats, 0, 0);
    	net.assignTargets(net.gmats, 0, 0);
    }
    
/**  Run the model forward given a state as input up to the action prediction layer. 
     Action selection/scoring layers are not updated.
     returns action predictions */
    def predict(states:FMat, nlayers:Int = 0) = {
    	val fstates = formatStates(states);
    	checkinit(fstates, null, null);
    	val nlayers0 = if (nlayers > 0) nlayers else (net.layers.length-1);
    	for (i <- 0 to nlayers0) net.layers(i).forward;
    }

/** Run the model all the way forward to the squared loss output layer, 
    and then backward to compute gradients.
    An action vector and reward vector must be given. */  

    def gradient(states:FMat, actions:IMat, rewards:FMat, ndout:Int=0) = {
      val ndout0 = if (ndout == 0) states.ncols else ndout;
    	val fstates = formatStates(states);
    	checkinit(fstates, actions, rewards);
    	net.forward;
    	net.setderiv(ndout0);
    	net.backward(0, 0);
    }

        
/** MSprop, i.e. RMSprop without the square root, or natural gradient */
        
    def msprop(learning_rate:Float) = {                
    	opts.lrate = learning_rate;
    	if (learning_rate > 1e-10f) {
    		adagrad.update(0,0,0);		
    		net.cleargrad;
    	}
    }

    def update_from(from_estimator:Estimator) = {
    	for (k  <- 0 until net.modelmats.length) {
    		net.modelmats(k) <-- from_estimator.net.modelmats(k);
    	}
    }
};

object Estimator {
  class Options extends Net.Options with ADAGrad.Opts {
    
  }
}
