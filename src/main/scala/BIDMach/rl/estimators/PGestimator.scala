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
  
  var invtemp:ConstantLayer = null;
  var entropyw:ConstantLayer = null;
  
  var preds:Layer = null;
  var probs:Layer = null;
  var advtgs:Layer = null;
  var entropy:Layer = null;
  var gain:Layer = null;
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
	  val conv1 =   conv(in)(w=7,h=7,nch=opts.nhidden,stride=4,pad=3,initv=1f,convType=opts.convType,hasBias=opts.hasBias);
	  val relu1 =   relu(conv1);
	  val conv2 =   conv(relu1)(w=3,h=3,nch=opts.nhidden2,stride=2,pad=0,convType=opts.convType,hasBias=opts.hasBias);
	  val relu2 =   relu(conv2);

	  // FC/reward prediction layers
	  val fc3 =     linear(relu2)(outdim=opts.nhidden3,initv=2e-2f,hasBias=opts.hasBias);
	  val relu3 =   relu(fc3);
	  preds =       linear(relu3)(outdim=opts.nactions,initv=5e-2f,hasBias=opts.hasBias); 

	  // Probability/ advantage layers
	  probs =       softmax(preds *@ invtemp); 
	  val pmean =   preds dot probs;
	  advtgs =      preds - pmean;

	  // Entropy layers
	  val logprobs = ln(probs + eps);
	  entropy =     (logprobs dot probs) *@ minus1;
	  nentropy =    Net.defaultLayerList.length;

	  // Action weighting
	  val aa =      advtgs(actions);
	  val apreds =  preds(actions);
	  val lpa =     logprobs(actions);
	  gain =        lpa *@ target;     
	  val weight =  fn2(target - apreds, aa)(fwdfn=weightedPGfn);

	  // Total weighted negloss, maximize this
	  val out =     lpa *@ weight + entropy *@ entropyw;

	  Net.getDefaultNet;
  }
	
	override val net = createNet;

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
     FMat(gain.output)
    		)    
  }
};

object PGestimator {
  trait Opts extends Estimator.Opts {
    var nhidden = 16;
    var nhidden2 = 32;
    var nhidden3 = 256;
    var nactions = 3;
  }
  
  class Options extends Opts {}
  
  def build(opts:Estimator.Opts) = {
    new PGestimator(opts.asInstanceOf[PGestimator.Opts])
  }
}
