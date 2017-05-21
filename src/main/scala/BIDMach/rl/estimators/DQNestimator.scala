package BIDMach.rl.estimators

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks.layers._;
import BIDMach._
import BIDMach.rl.algorithms._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import BIDMach.networks._


class DQNestimator(opts:DQNestimator.Options = new DQNestimator.Options) extends Estimator {
  
  var invtemp:ConstantLayer = null;
  var entropyw:ConstantLayer = null;
  
  var preds:Layer = null;
  var probs:Layer = null;
  var entropy:Layer = null;
  var loss:Layer = null;
  var nentropy = 0;
    
	def createNet:Net = {
	  import BIDMach.networks.layers.Layer._;
	  Net.initDefault(opts);

	  // Input layers 
	  val in =      input;
	  val actions = input;
	  val target =  input;
	  
	  // Settable param layers;
	  invtemp  =    const(1f);
	  entropyw=     const(1f);

	  // Random constants
	  val minus1 =  const(-1f);
	  val eps =     const(1e-6f);

	  // Convolution layers
	  val conv1 =   conv(in)(w=7,h=7,nch=opts.nhidden,stride=4,pad=3,initv=1f,convType=opts.convType);
	  val relu1 =   relu(conv1);
	  val conv2 =   conv(relu1)(w=3,h=3,nch=opts.nhidden2,stride=2,pad=0,convType=opts.convType);
	  val relu2 =   relu(conv2);

	  // FC/reward prediction layers
	  val fc3 =     linear(relu2)(outdim=opts.nhidden3,initv=2e-2f);
	  val relu3 =   relu(fc3);
	  preds =       linear(relu3)(outdim=opts.nactions,initv=5e-2f); 

	  // Probabilitylayers
	  probs =       softmax(preds *@ invtemp); 

	  // Entropy layers
	  entropy =     (ln(probs + eps) dot probs) *@ minus1;
	  nentropy =    Net.defaultLayerList.length;

	  // Action loss layers
	  val diff =    target - preds(actions);
	  loss =        diff *@ diff;                     // Index of base loss layer.

	  // Total weighted negloss, maximize this
	  val out =     loss *@ minus1 + entropy *@ entropyw;

	  Net.getDefaultNet;
  }

	// Set temperature and entropy weight
  override def setConsts2(invtemperature:Float, entropyWeight:Float) = {
	  invtemp.opts.value =  invtemperature;
	  entropyw.opts.value =  entropyWeight;
  }
  
  // Get the Q-predictions, action probabilities, entropy and loss for the last forward pass. 
  override def getOutputs4:(FMat,FMat,FMat,FMat) = {
    (FMat(preds.output),
     FMat(probs.output),
     FMat(entropy.output),
     FMat(loss.output)
    		)    
  }
};

object DQNestimator {
  class Options extends NDQNalgorithm.Options {
    var nhidden = 16;
    var nhidden2 = 32;
    var nhidden3 = 256;
    var nactions = 3;
  }
}
