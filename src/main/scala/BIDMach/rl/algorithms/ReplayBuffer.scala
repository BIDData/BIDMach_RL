package BIDMach.rl.algorithms;

import BIDMat.{Mat,SBMat,BMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
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
import scala.util.hashing.MurmurHash3;
import java.util.HashMap;

class ReplayBuffer(obs:Mat, action:Mat, length:Int) { 
  var nitems = 0;
  var nexti = 0;
  val dims = obs.dims
  val observations:Mat = obs match { 
    case bobs:BMat => bzeros(dims \ length)
    case iobs:IMat => izeros(dims \ length)
    case fobs:FMat => zeros(dims \ length)
  }
  val adims = action.dims
  val actions:Mat = action match { 
    case bobs:BMat => bzeros(adims \ length)
    case iobs:IMat => izeros(adims \ length)
    case fobs:FMat => zeros(adims \ length)
  }
  val rewards = zeros(1, length);
  val dones = izeros(1, length);

  def push(action:Mat, obs:Mat, reward:Float=0, done:Int= 0) = { 
    action.reshapeView(action.dims\1).colslice(0,1,actions,nexti);
    obs.reshapeView(obs.dims\1).colslice(0,1,observations,nexti);
    rewards(0,nexti) = reward;
    dones(0,nexti) = done;
    nexti = (nexti + 1) % length
    nitems = math.min(nitems + 1, length);
    this
  }

  def sample(n:Int, obs_per_state:Int = 1, oobs:Mat = null, oactions:Mat = null, orewards:FMat = null, odones:IMat = null):(Mat, Mat, FMat, IMat) = { 
    val obsdims = dims \ (obs_per_state * n);
    val obsdims2 = dims \ obs_per_state \ n;
    val actdims = adims \ n;
    val obs = if (oobs.asInstanceOf[AnyRef] != null) oobs else observations.zeros(obsdims);
    val action = if (oactions.asInstanceOf[AnyRef] != null) oactions else actions.zeros(actdims);
    val reward = if (orewards.asInstanceOf[AnyRef] != null) orewards else zeros(1, n);
    val done = if (odones.asInstanceOf[AnyRef] != null) odones else izeros(1, n);
    val irand = min(length - 1, int(rand(1, n)*length))
    val instate = observations.zeros(dims \ obs_per_state);
    val outstate = obs.zeros(dims \ obs_per_state);
    for (i <- 0 until n) { 
      observations.colslice(irand(i)*obs_per_state, (irand(i)+1)*obs_per_state, instate, 0);
      outstate <-- instate;
      outstate.colslice(0, obs_per_state, obs, i*obs_per_state);
      actions.colslice(irand(i), (irand(i)+1), action, i);
      reward(0, i) = rewards(0, irand(i));
      done(0, i) = dones(0, irand(i));
    }
    (obs.reshapeView(obsdims2), action, reward, done)
  }
  
}
