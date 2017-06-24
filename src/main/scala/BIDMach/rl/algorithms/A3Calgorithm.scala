package BIDMach.rl.algorithms;

import BIDMat.{Mat,SBMat,CMat,DMat,FMat,IMat,LMat,HMat,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,SMat,SDMat}
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

class A3Calgorithm(
		val envs:Array[Environment], 
		val parstepper:(Array[Environment], IMat, Array[FMat], FMat, FMat) => (Array[FMat], FMat, FMat),
		val buildEstimator:(Estimator.Opts) => Estimator,
		val opts:A3Calgorithm.Opts = new A3Calgorithm.Options
		) extends Algorithm(opts) {
  
	val npar = envs.length;                            // Number of parallel environments 
	
	var VALID_ACTIONS:IMat = null;
	var nactions = 0;
	var obs0:FMat = null;
	var total_steps = 0;
	var block_reward = 0f;
	var total_reward = 0f;
	var block_entropy = 0f;
	var block_loss = 0f;
	var block_count = 0;
	var total_epochs = 0;
	var total_time = 0f;
	var rbaseline = 0f;
	var rbaseline0 = 0f;
	var igame = 0;
	var state:FMat = null;
	val rn = new java.util.Random;
	
	var save_length = 10000;
	var saved_frames:FMat = null;
	var saved_actions:IMat = null;
	var saved_preds:FMat = null; 
	var reward_plot:FMat = null;
	val dtimes = zeros(1,7);
	
	var estimator:Estimator = null;
	
	def startup {
	  tic;
	  obs0 = null;
	  total_steps = 0;
	  block_reward = 0f;
	  block_count = 0;
	  total_epochs = 0;
	  VALID_ACTIONS = envs(0).VALID_ACTIONS;
	  nactions = VALID_ACTIONS.length;
	  val nwindow = opts.nwindow;
	  state = zeros(envs(0).statedims\nwindow\npar);
	  
	  save_length = opts.save_length;
	  saved_frames = zeros(envs(0).statedims\save_length);
	  saved_actions = izeros(1,save_length);
	  saved_preds = zeros(1,save_length);
	  
	  print("Initializing Environments")
	  for (i <- 0 until npar) {
	  	val nmoves = rn.nextInt(opts.init_moves - nwindow) + nwindow;
	  	for (j <- 0 until nmoves) {   
	  		val action = VALID_ACTIONS(rn.nextInt(nactions));
	  		val (obs, reward, done) = envs(i).step(action);
	  		obs0 = obs;
	  		total_steps += 1;
	  		if (nmoves - j <= nwindow) {
	  			val k = nwindow - nmoves + j;
	  			state(?,?,k,i) = obs;
	  		}
	  		if (done || reward != 0) {
	  			block_reward += reward;
	  			block_count += 1;
	  		}
	  		if (done) {
	  			envs(i).reset();
	  			total_epochs += 1;
	  		}
	  	}
	  	print(".");
	  }

	  rbaseline = block_reward/block_count.toFloat;
	  rbaseline0 = rbaseline;
	  total_time = toc;     
	  println("\n%d steps, %d epochs in %5.4f seconds at %5.4f msecs/step" format(
	  		total_steps, total_epochs, total_time, 1000f*total_time/total_steps))

	}
  
  def train {
    val nsteps = opts.nsteps;
    val nwindow = opts.nwindow;
    val learning_rates = loginterp(opts.lr_schedule, nsteps+1);
    val temperatures = loginterp(opts.temp_schedule, nsteps+1);
    val ndqn = opts.ndqn;
    
    total_steps = 0;
    block_reward = 0f;
    total_reward = 0f;
    block_entropy = 0f;
    block_loss = 0f;
    block_count = 0;
    total_epochs = 0;
    total_time = 0f;
  	igame = 0;
  	
  	var last_epochs = 0;
  	val new_state = state.copy;
  	var dobaseline = false;
  	val baselinethresh  = 0.1f;
  	val GPUcacheState = Mat.useGPUcache;
  	val cacheState = Mat.useCache;
  	Mat.useGPUcache = true;
  	Mat.useCache = false;
    
// Create estimators
  	estimator = buildEstimator(opts.asInstanceOf[Estimator.Opts]);
  	estimator.predict4(state);    //	Initialize them by making predictions
  	  	
  	val times = zeros(1,8);
  	dtimes.clear;
  	val ractions = int(rand(1, npar) * nactions);
  	val (obs0, rewards0, dones0) = parstepper(envs, VALID_ACTIONS(ractions), null, null, null);           // step through parallel envs
 
  	var actions = izeros(1,npar);
  	var action_probs:FMat = null;
  	val rand_actions = ones(nactions, npar) * (1f/nactions); 
  	val printsteps0 = opts.print_steps / ndqn * ndqn; 
  	  	
  	val state_memory = zeros(envs(0).statedims\(opts.nwindow+ndqn-1)\npar);
  	val action_memory = izeros(ndqn, npar);
  	val reward_memory = zeros(ndqn+1, npar);
  	val done_memory = zeros(ndqn+1, npar);
  	reward_plot = zeros(1, nsteps/printsteps0);

  	tic;
  	for (istep <- ndqn to opts.nsteps by ndqn) {
//    if (render): envs[0].render()
  		val lr = learning_rates(istep);                                // update the decayed learning rate
  		val temp = temperatures(istep);                                // get an epsilon for the eps-greedy policy
  		estimator.setConsts3(1/temp, opts.entropy_weight, opts.policygrad_weight);
  		
  		reward_memory(0,?) = reward_memory(ndqn,?);
  		done_memory(0,?) = done_memory(ndqn,?);
  		for (i <- 0 until ndqn) {
  			times(0) = toc;
  			estimator.predict4(state); // get the next action probabilities etc from the policy
  			val (preds, aprobs, _, _) = estimator.getOutputs4;
  			times(1) = toc;

  			actions <-- multirnd(aprobs);                                              // Choose actions using the policy 
  			val (obs, rewards, dones) = parstepper(envs, VALID_ACTIONS(actions), obs0, rewards0, dones0);           // step through parallel envs
  			times(2) = toc;

  			for (j <- 0 until npar) {                                                                                                    // process the observation
  				new_state(?,?,0->(nwindow-1),j) = state(?,?,1->nwindow,j);              // shift the image stack and add a new image
  				new_state(?,?,nwindow-1,j) = obs(j);         
  			}
  			saved_frames(?,?,igame) = obs(0).reshapeView(envs(0).statedims\1);
  			saved_actions(0,igame) = actions(0);
  			saved_preds(0,igame) = preds(0);
  			igame = (igame+1) % save_length;
  			
  			total_epochs += sum(dones).v.toInt;
  			block_reward += sum(rewards).v;

  			times(3) = toc;

  			dones <-- (dones + (rewards != 0f) > 0f);

  			if (sum(dones).v > 0) rbaseline = opts.baseline_decay * rbaseline + (1-opts.baseline_decay) * (sum(rewards).v / sum(dones).v);
  			if (! dobaseline && rbaseline - rbaseline0 > baselinethresh * (envs(0).score_range(1) - envs(0).score_range(0))) {
  				dobaseline = true;
  				rbaseline0 = rbaseline;
  			}
  			val arewards = if (dobaseline) {
  				rewards - (dones > 0) *@ (rbaseline - rbaseline0);
  			} else {
  				rewards;
  			}
  			if (i == 0) {
  			  state_memory(?,?,0->nwindow,?) = state;
  			} else {
  			  state_memory(?,?,nwindow+i-1,?) = state(?,?,nwindow-1,?);
  			}
  			action_memory(i,?) = actions;
  			reward_memory(i+1,?) = arewards;
  			done_memory(i+1,?) = dones;
  			state <-- new_state;
  			times(4) = toc;
  			dtimes(0,0->4) = dtimes(0,0->4) + (times(0,1->5) - times(0,0->4));
  		}
  		estimator.predict4(new_state);
  		val (v_next, _, _, _) = estimator.getOutputs4; 
  		times(5) = toc;

  		reward_memory(ndqn,?) = done_memory(ndqn,?) *@ reward_memory(ndqn,?) + (1f-done_memory(ndqn,?)) *@ v_next; // Add to reward mem if no actual reward
  		for (i <- (ndqn-1) to 0 by -1) {
  			// Propagate rewards back in time. Actual rewards override predicted rewards. 
  			reward_memory(i,?) = done_memory(i,?) *@ reward_memory(i,?) + (1f - done_memory(i,?)) *@ reward_memory(i+1,?) *@ opts.discount_factor;
  		}

  		// Now compute gradients for the states/actions/rewards saved in the table.
  		for (i <- 0 until ndqn) {
  			new_state <-- state_memory(?,?,i->(i+nwindow),?);
  			estimator.gradient4(new_state, action_memory(i,?), reward_memory(i,?), reward_memory(i+1,?));
  			val (_, _, ev, lv) = estimator.getOutputs4;
  			block_loss += sum(lv).v;                                // compute q-estimator gradient and return the loss
  			block_entropy += sum(ev).v; 
  		}
  		times(6) = toc;

  		estimator.msprop(lr);                       // apply the gradient update
  		times(7) = toc;

  		dtimes(0,4->7) = dtimes(0,4->7) + (times(0,5->8) - times(0,4->7));
  		val t = toc;
  		if (istep % printsteps0 == 0) {
  			total_reward += block_reward;
  			println("I %5d, T %4.1f, L %7.6f, Ent %5.4f, E %d, R/E %5.4f, CR/E %5.4f" 
  					format(istep, t, block_loss/printsteps0/npar, block_entropy/printsteps0/npar, 
  							total_epochs, block_reward/math.max(1,total_epochs-last_epochs), total_reward/math.max(1,total_epochs)));
  			reward_plot(istep/printsteps0-1) = block_reward/math.max(1,total_epochs-last_epochs);
  			last_epochs = total_epochs;
  			block_reward = 0f;
  			block_loss = 0f;
  			block_entropy = 0f;
  		}
  	}
  	Mat.useGPUcache = GPUcacheState;
  	Mat.useCache = cacheState;
  }
}

object A3Calgorithm {
  trait Opts extends Algorithm.Opts {
    
    var nsteps = 400000;                             // Number of steps to run (game actions per environment)
  	var ndqn = 5;                                    // Number of DQN steps per update
  	var print_steps = 10000;                         // Number of steps between printouts
  	var init_moves = 4000;                           // Upper bound on random number of moves to take initially
  	var nwindow = 4;                                 // Sensing window = last n images in a state
  	var save_length = 10000;
  	
  	var discount_factor = 0.99f;                     // Reward discount factor
  	var policygrad_weight = 0.3f;                    // Weight of policy gradient compared to regression loss
  	var entropy_weight = 1e-4f;                      // Entropy regularization weight
  	var baseline_decay = 0.9999f;                    // Reward baseline decay
  	
  	var lr_schedule = (0f \ 3e-6f on 1f \ 3e-6f);    // Learning rate schedule
  	var temp_schedule = (0f \ 1f on 1f \ 1f);        // Temperature schedule
  }
  
  class Options extends Opts {}
}