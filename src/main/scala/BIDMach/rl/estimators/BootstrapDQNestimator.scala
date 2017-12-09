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


class BootstrapDQNestimator(val opts:BootstrapDQNestimator.Opts = new BootstrapDQNestimator.Options) extends Estimator {
	
  var predsLayer:Layer = null;
  var lossLayer:Layer = null;
  
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
	  val in =      input();
	  val actions = input();
	  val target =  input();
	  val bootsample = input();

	  // Random constants
	  val minus1 =  const(-1f);

	  // Convolution layers
	  val conv1 =       conv(in)(w=8,h=8,nch=opts.nhidden,stride=4,pad=0,hasBias=opts.hasBias);
	  val relu1 =       relu(conv1)(inplace=opts.inplace);
	  val conv2 =       conv(relu1)(w=4,h=4,nch=opts.nhidden2,stride=2,pad=0,hasBias=opts.hasBias);
	  val relu2 =       relu(conv2)(inplace=opts.inplace);
	  
	  val allpreds =    new Array[NodeTerm](opts.ntails);
	  val alllosses =   new Array[NodeTerm](opts.ntails);
	  
	  for (i <- 0 until opts.ntails) {
	  // FC/reward prediction layers
	  	val fc3 =       linear(relu2)(outdim=opts.nhidden3,hasBias=opts.hasBias);
	  	val relu3 =     relu(fc3)(inplace=opts.inplace);
	  	val preds =     linear(relu3)(outdim=opts.nactions,hasBias=opts.hasBias); 
	  	// Action loss layers
	  	val diff =      target - preds(actions);
	  	val loss =      diff *@ diff;                     // Base loss layer.
	  	allpreds(i) =   preds;
	  	alllosses(i) =  loss;
	  }
	  val preds =       stack(allpreds);
	  val stackloss =   stack(alllosses);
	  val wloss =       stackloss *@ bootsample;
	  val loss =        sum(wloss);
	  
	  // Total weighted negloss, maximize this
	  val out =         loss *@ minus1 

	  opts.nodeset = Net.getDefaultNodeSet;
	  
	  val net = new Net(opts);
	  
	  net.createLayers;
	  
	  predsLayer = preds.myLayer;
	  lossLayer = loss.myLayer;
	  
	  net;
  }

  override val net = createNet;
  
  // Get the Q-predictions and loss for the last forward pass. 
  override def getOutputs2:(FMat,FMat) = {
    (FMat(predsLayer.output),
     FMat(lossLayer.output)
    		)    
  }
};

object BootstrapDQNestimator {
  trait Opts extends DQNestimator.Opts {
    var ntails = 8;
  }
  
  class Options extends Opts {}
  
  def build(opts:Estimator.Opts) = {
    new BootstrapDQNestimator(opts.asInstanceOf[BootstrapDQNestimator.Opts])
  }
}
