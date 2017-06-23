package BIDMach.rl.estimators

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach._
import BIDMach.networks._
import BIDMach.networks.layers._;
import BIDMach.rl.algorithms._
import jcuda.jcudnn._
import jcuda.jcudnn.JCudnn._
import edu.berkeley.bid.MurmurHash3.MurmurHash3_x64_64;


class DQNestimator(opts:DQNestimator.Opts = new DQNestimator.Options) extends Estimator {
  
  var invtemp:ConstantNode = null;
  var entropyw:ConstantNode = null;
  
  var predsLayer:Layer = null;
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
	  val target =  input;
	  
	  // Settable param layers;
	  invtemp  =    const(1f);
	  entropyw=     const(1f);

	  // Random constants
	  val minus1 =  const(-1f);
	  val eps =     const(1e-6f);

	  // Convolution layers
	  val conv1 =   conv(in)(w=7,h=7,nch=opts.nhidden,stride=4,pad=3,initv=1f,convType=opts.convType,hasBias=opts.hasBias);
	  val relu1 =   relu(conv1)(inplace);
	  val conv2 =   conv(relu1)(w=3,h=3,nch=opts.nhidden2,stride=2,pad=0,convType=opts.convType,hasBias=opts.hasBias);
	  val relu2 =   relu(conv2)(inplace);

	  // FC/reward prediction layers
	  val fc3 =     linear(relu2)(outdim=opts.nhidden3,initv=2e-2f,hasBias=opts.hasBias);
	  val relu3 =   relu(fc3)(inplace);
	  val preds =   linear(relu3)(outdim=opts.nactions,initv=5e-2f,hasBias=opts.hasBias); 

	  // Probabilitylayers
	  val probs =   softmax(preds *@ invtemp); 

	  // Entropy layers
	  val entropy = (ln(probs + eps) dot probs) *@ minus1;
	  val nentropy= Net.defaultNodeList.length;

	  // Action loss layers
	  val diff =    target - preds(actions);
	  val loss =    diff *@ diff;                     // Base loss layer.

	  // Total weighted negloss, maximize this
	  val out =     loss *@ minus1 + entropy *@ entropyw;

	  opts.nodeset = Net.getDefaultNodeSet;
	  
	  val net = new Net(opts);
	  
	  net.createLayers;
	  
	  predsLayer = preds.myLayer;
	  probsLayer = probs.myLayer;
	  entropyLayer = entropy.myLayer;
	  lossLayer = loss.myLayer;
	  
	  net;
  }

  override val net = createNet;


	// Set temperature and entropy weight
  override def setConsts2(invtemperature:Float, entropyWeight:Float) = {
	  invtemp.value =  invtemperature;
	  entropyw.value =  entropyWeight;
  }
  
  // Get the Q-predictions, action probabilities, entropy and loss for the last forward pass. 
  override def getOutputs4:(FMat,FMat,FMat,FMat) = {
    (FMat(predsLayer.output),
     FMat(probsLayer.output),
     FMat(entropyLayer.output),
     FMat(lossLayer.output)
    		)    
  }
};

object DQNestimator {
  trait Opts extends Estimator.Opts {
    var nhidden = 16;
    var nhidden2 = 32;
    var nhidden3 = 256;
    var nactions = 3;
  }
  
  class Options extends Opts {}
  
  def build(opts:Estimator.Opts) = {
    new DQNestimator(opts.asInstanceOf[DQNestimator.Opts])
  }
}
