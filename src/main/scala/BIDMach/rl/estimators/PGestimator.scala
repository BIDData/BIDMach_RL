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


class PGestimator(opts:PGestimator.Opts = new PGestimator.Options) extends Estimator {
  
  var temp:ConstantNode = null;
  var entropyw:ConstantNode = null;
  
  var predsLayer:Layer = null;
  var probsLayer:Layer = null;
  var advtgsLayer:Layer = null;
  var entropyLayer:Layer = null;
  var gainLayer:Layer = null;
  var nentropy = 0;

  override def formatStates(s:FMat) = {
    if (net.opts.tensorFormat == Net.TensorNCHW) {
    	s.reshapeView(s.dims(2), s.dims(0), s.dims(1), s.dims(3));
    } else {
    	val x = s.transpose(2\0\1\3);
    	x.setGUID(MurmurHash3_x64_64(Array(s.GUID), "transpose213".##));
    	x;
    }
  }
  
  def weightedPG(d:Mat, a:Mat):Mat = {
    val s = sign(d);
    s *@ sqrt(abs(d *@ a));
  }
  
  val weightedPGfn = weightedPG _;
    
  def createNet = {
  	import BIDMach.networks.layers.Node._;
  	Net.initDefaultNodeSet;

  	// Input layers 
  	val in =      input();
  	val actions = input();
  	val target =  input();

  	// Settable param layers;
  	val temp  =   const(1f)();
  	val entropyw= const(1f)();

  	// Random constants
  	val minus1 =  const(-1f)();
  	val eps =     const(1e-2f)();

  	// Convolution layers
  	val conv1 =   conv(in)(w=8,h=8,nch=opts.nhidden,stride=4,pad=0,hasBias=opts.hasBias);
  	val relu1 =   relu(conv1)(inplace=opts.inplace);
  	val conv2 =   conv(relu1)(w=4,h=4,nch=opts.nhidden2,stride=2,pad=0,hasBias=opts.hasBias);
  	val relu2 =   relu(conv2)(inplace=opts.inplace);

  	// FC/reward prediction layers
  	val fc3 =     linear(relu2)(outdim=opts.nhidden3,hasBias=opts.hasBias);
  	val relu3 =   relu(fc3)(inplace=opts.inplace);
  	val preds =   linear(relu3)(outdim=opts.nactions,hasBias=opts.hasBias); 

  	// Probability/ advantage layers
  	val probs =   softmax(preds / temp); 
  	val pmean =   preds dot probs;
  	val advtgs =  preds - pmean;

  	// Entropy layers
  	val logprobs = ln(probs + eps);
  	val entropy =  (logprobs dot probs) *@ minus1;
  	val nentropy=  Net.defaultNodeList.length;

  	// Action weighting
  	val aa =      advtgs(actions);
  	val apreds =  preds(actions);
  	val lpa =     logprobs(actions) *@ temp;  
  	//	  val weight =  fn2(target - apreds, aa)(fwdfn=weightedPGfn);
  	val weight =  target - apreds;
  	val gain =    weight *@ weight;

  	// Total weighted negloss, maximize this
  	val out =     lpa *@ forward(weight) + entropy *@ entropyw;

  	opts.nodeset = Net.getDefaultNodeSet;

  	val net = new Net(opts);

  	net.createLayers;

  	predsLayer = preds.myLayer;
  	probsLayer = probs.myLayer;
  	advtgsLayer = advtgs.myLayer;
  	entropyLayer = entropy.myLayer;
  	gainLayer = gain.myLayer;

  	net;
  }

  override val net = createNet;

	// Set temperature and entropy weight
  override def setConsts2(temperature:Float, entropyWeight:Float) = {
	  temp.value =  temperature;
	  entropyw.value =  entropyWeight;
  }
  
  // Get the Q-predictions, action probabilities, entropy and loss for the last forward pass. 
  override def getOutputs4:(FMat,FMat,FMat,FMat) = {
    (FMat(predsLayer.output),
     FMat(probsLayer.output),
     FMat(entropyLayer.output),
     FMat(gainLayer.output)
    		)    
  }
};

object PGestimator {
  trait Opts extends Estimator.Opts {
    var nhidden = 16;
    var nhidden2 = 32;
    var nhidden3 = 256;
    var nactions = 3;
    tensorFormat = Net.TensorNCHW;
  }
  
  class Options extends Opts {}
  
  def build(opts:Estimator.Opts) = {
    new PGestimator(opts.asInstanceOf[PGestimator.Opts])
  }
}
