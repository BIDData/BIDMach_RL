package BIDMach.rl.estimators

//
// An V-critic (standard) Estimator for A3C. 
// First input is the current state. 
// Second input is the action taken by the policy. 
// Third input is the target value of the current state.
// Fourth input is the target value of the next state (after the action was applied). 
//

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.networks.layers._;
import BIDMach.networks._
import BIDMach._
import BIDMach.rl.algorithms._;
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import edu.berkeley.bid.MurmurHash3.MurmurHash3_x64_64;


class A3CestimatorV(opts:A3CestimatorV.Opts = new A3CestimatorV.Options) extends Estimator {
  
  var invtemp:ConstantNode = null;
  var entropyw:ConstantNode = null;
  var gradw:ConstantNode = null;
  
  var vpredsLayer:Layer = null;
  var probsLayer:Layer = null;
  var entropyLayer:Layer = null;
  var lossLayer:Layer = null;
  var nentropy = 0;

  val inplace = true;
  
  override def formatStates(s:FMat) = {
    if (net.opts.tensorFormat == Net.TensorNCHW) {
    	s.reshapeView(s.dims(2), s.dims(0), s.dims(1), s.dims(3));
    } else {
    	val x = s.transpose(2\0\1\3);
    	x.setGUID(MurmurHash3_x64_64(Array(s.GUID), "transpose213".##));
    	x;
    }
  }
    
  def createNet = { 
	  import BIDMach.networks.layers.Node._;
	  Net.initDefaultNodeSet;

	  // Input layers 
	  val in =      input;
	  val actions = input;
	  val vtarget =  input;
	  val atarget =  input;
	  
	  // Settable param layers;
	  invtemp  =    const(1f);
	  entropyw =    const(1f);
	  gradw =       const(1f);

	  // Random constants
	  val minus1 =  const(-1f);
	  val eps =     const(1e-6f);

	  // Convolution layers
	  val conv1 =   conv(in)(w=7,h=7,nch=opts.nhidden,stride=4,pad=3);
	  val relu1 =   relu(conv1)(inplace);
	  val conv2 =   conv(relu1)(w=3,h=3,nch=opts.nhidden2,stride=2,pad=0);
	  val relu2 =   relu(conv2)(inplace);

	  // FC/reward prediction layers
	  val fc3 =     linear(relu2)(outdim=opts.nhidden3);
	  val relu3 =   relu(fc3)(inplace); 
	  val ppreds =  linear(relu3)(outdim=opts.nactions);
	  val vpreds =  linear(relu3)(outdim=1);

	  // Probability layers
	  val probs =   softmax(ppreds *@ invtemp); 

	  // Entropy layers
	  val logprobs= ln(probs + eps);
	  val entropy = (logprobs dot probs) *@ minus1;
	  val nentropy= Net.defaultNodeList.length;

	  // Value loss layers
	  val diff =    vtarget - vpreds;
	  val loss =    diff *@ diff;     
	  
	  // Policy gradient
	  val advtg =   atarget - vpreds;
	  val pgrad =   logprobs(actions) *@ forward(advtg);

	  // Total weighted negloss, maximize this
	  val out =     loss *@ minus1 + pgrad *@ gradw + entropy *@ entropyw;

	  opts.nodeset = Net.getDefaultNodeSet;
	  
	  val net = new Net(opts);
	  
	  net.createLayers;
	  
	  vpredsLayer = vpreds.myLayer;
    probsLayer = probs.myLayer;
    entropyLayer = entropy.myLayer;
    lossLayer = loss.myLayer;
    
	  net
  }

  override val net = createNet;

	// Set temperature and entropy weight
  override def setConsts3(invtemperature:Float, entropyWeight:Float, gradWeight:Float) = {
	  invtemp.value =  invtemperature;
	  entropyw.value = entropyWeight;
	  gradw.value =    gradWeight;
  }
  
  // Get the Q-predictions, action probabilities, entropy and loss for the last forward pass. 
  override def getOutputs4:(FMat,FMat,FMat,FMat) = {
    (FMat(vpredsLayer.output),
     FMat(probsLayer.output),
     FMat(entropyLayer.output),
     FMat(lossLayer.output)
    		)    
  }
};

object A3CestimatorV {
  trait Opts extends Estimator.Opts {
    var nhidden = 16;
    var nhidden2 = 32;
    var nhidden3 = 256;
    var nactions = 3;
  }
  
  class Options extends Opts {}
  
  def build(opts:Estimator.Opts) = {
    new A3CestimatorV(opts.asInstanceOf[A3CestimatorV.Opts])
  }
}
