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
  var bootsample:FMat = null;
  val rn = new java.util.Random;
  
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
//	  val scaleHead = 1f/math.sqrt(opts.ntails).toFloat;
	  val scaleHead = 1f/opts.ntails;
	  

	  // Input layers 
	  val in =          input();
	  val actions =     input();
	  val target =      input();
	  val bootsample =  input();

	  // Constants
	  val minus1 =      const(-1f);
	  val invntails =   const(1f/opts.ntails);

	  // Convolution layers
	  val conv1 =       conv(in)(w=8,h=8,nch=opts.nhidden,stride=4,pad=0,hasBias=opts.hasBias,
			                         lr_scale=scaleHead,bias_scale=scaleHead);
	  val relu1 =       relu(conv1)(inplace=opts.inplace);
	  val conv2 =       conv(relu1)(w=4,h=4,nch=opts.nhidden2,stride=2,pad=0,hasBias=opts.hasBias,
	                                lr_scale=scaleHead,bias_scale=scaleHead);
	  val relu2 =       relu(conv2)(inplace=opts.inplace);

	  // FC/reward prediction layers
	  val fc3 =         linear(relu2)(outdim=opts.nhidden3*opts.ntails,hasBias=opts.hasBias);
	  val relu3 =       relu(fc3)(inplace=opts.inplace);
	  val preds =       linear(relu3)(outdim=opts.nactions*opts.ntails,hasBias=opts.hasBias,ngroups=opts.ntails); 
	  
	  // Action loss layers
	  val diff =        preds(actions) - target;
	  val stackloss =   diff *@ diff;                     // Base loss layer.
	  
	  // Apply bootstrap sample weights to the losses
	  val tloss =       sum(stackloss *@ bootsample);
	  val loss =        tloss *@ invntails;
	  
	  // Total weighted negloss, maximize this
	  val out =         tloss *@ minus1 

	  opts.nodeset = Net.getDefaultNodeSet;
	  
	  val net = new Net(opts);
	  
	  net.createLayers;
	  
	  predsLayer = preds.myLayer;
	  lossLayer = loss.myLayer;
	  
	  net;
  }

  override val net = createNet;
  
  override def predict(states:FMat, nlayers:Int = 0) = {
  	val fstates = formatStates(states);
  	if (bootsample.asInstanceOf[AnyRef] == null) bootsample = zeros(opts.ntails, states.ncols);
  	bootsample.set(1f/opts.ntails);
  	checkinit(fstates, izeros(opts.ntails, fstates.ncols), zeros(opts.ntails, fstates.ncols), bootsample); 	
  	val nlayers0 = if (nlayers > 0) nlayers else (net.layers.length-1);
  	for (i <- 0 to nlayers0) net.layers(i).forward;
  }
  
  override def getOutputs2:(FMat,FMat) = {
		val q_vals = FMat(predsLayer.output);
		val loss = FMat(lossLayer.output);
    (q_vals, loss);
  }
  
  // Compute gradient by applying a poisson random bootstrap weight
  override def gradient(states:FMat, actions:IMat, rewards:FMat, npar:Int):Unit = {
    bootsample <-- poissrnd(ones(opts.ntails, npar));
    gradient4(states, actions, rewards, bootsample, npar);
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
