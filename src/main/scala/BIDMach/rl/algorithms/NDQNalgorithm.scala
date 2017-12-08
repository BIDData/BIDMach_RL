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

class NDQNalgorithm(
		val envs:Array[Environment], 
		val parstepper:(Array[Environment], IMat, Array[FMat], FMat, FMat) => (Array[FMat], FMat, FMat),
		val buildEstimator:(Estimator.Opts) => Estimator,
		val opts:NDQNalgorithm.Opts = new NDQNalgorithm.Options
		) extends Algorithm(opts) {
  
	val npar = envs.length;                            // Number of parallel environments 
	
	var VALID_ACTIONS:IMat = null;
	var nactions = 0;
	var total_steps = 0;
	var block_reward = 0f;
	var total_reward = 0f;
	var block_loss = 0f;
	var block_entropy = 0f;
	var block_count = 0;
	var total_epochs = 0;
	var total_time = 0f;
	var igame = 0;
	var state:FMat = null;
	var zstate:FMat = null;
	var mean_state:FMat = null;
	var obs0:FMat = null;
	val rn = new java.util.Random;
	var ndqn_max:Int = 0;
	var xhist:FMat = null;
	
	var q_estimator:Estimator = null;
	var t_estimator:Estimator = null;
	val dtimes = zeros(1,7);
	val times = zeros(1,8);
	
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
	  zstate = zeros(envs(0).statedims\nwindow\npar);
	  mean_state = zeros(envs(0).statedims\1\1);
	  
	  save_length = opts.save_length;
	  saved_frames = zeros(envs(0).statedims\save_length);
	  saved_actions = izeros(1, save_length);
	  saved_rewards = zeros(1, save_length);
	  saved_dones = zeros(1, save_length);
	  saved_lives = zeros(1, save_length);
	  saved_preds = zeros(nactions, save_length);
	  ndqn_max = if (opts.ndqn_max == 0) opts.ndqn * 4 else opts.ndqn_max;
	  xhist = zeros(1, ndqn_max);
	  
	  print("Initializing Environments")
	  for (i <- 0 until npar) {
	  	val nmoves = rn.nextInt(opts.init_moves - nwindow) + nwindow;
	  	for (j <- 0 until nmoves) {   
	  		val action = VALID_ACTIONS(rn.nextInt(nactions));
	  		val (obs, reward, done) = envs(i).step(action);
	  		obs0 = obs;
	  		total_steps += 1;
	  		mean_state = mean_state + obs.reshapeView(obs.dims(0), obs.dims(1), 1, 1);
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
	  mean_state ~ mean_state / total_steps;
	  total_time = toc;     
	  println("\n%d steps, %d epochs in %5.4f seconds at %5.4f msecs/step" format(
	  		total_steps, total_epochs, total_time, 1000f*total_time/total_steps))

	}
  
  def train {
    val nsteps = opts.nsteps;
    val nwindow = opts.nwindow;
    val ndqn = opts.ndqn;
    val old_lives = zeros(1, npar);
    val new_lives  = zeros(1, npar);
    val u0 = math.pow(1 - 1.0/ndqn, ndqn_max);
    val v0 = math.log(1 - 1.0/ndqn);
    val irange = irow(opts.nexact->npar);
    
    total_steps = 0;
    block_reward = 0f;
    total_reward = 0f;
    block_loss = 0f;
    block_entropy = 0f;
    block_count = 0;
    total_epochs = 0;
    total_time = 0f;
  	igame = 0;
  	
  	var last_epochs = 0;
  	val new_state = state.copy;
  	val GPUcacheState = Mat.useGPUcache;
  	val cacheState = Mat.useCache;
  	Mat.useGPUcache = true;
  	Mat.useCache = false;
    
// Create estimators
  	q_estimator = buildEstimator(opts.asInstanceOf[Estimator.Opts]);
  	t_estimator = buildEstimator(opts.asInstanceOf[Estimator.Opts]);
  	q_estimator.predict(state);    //	Initialize them by making predictions
  	t_estimator.predict(state);

  	dtimes.clear;
  	val ractions = int(rand(1, npar) * nactions);
  	val (obs0, rewards0, dones0) = parstepper(envs, VALID_ACTIONS(ractions), null, null, null);           // step through parallel envs
 
  	var actions = izeros(1,npar);
  	var action_probs:FMat = null;
  	val rand_actions = ones(nactions, npar) * (1f/nactions);
  	val targwin = math.max(opts.target_window, ndqn_max*2);
  	val printsteps0 = opts.print_steps;
  	  	
  	val state_memory = zeros(envs(0).statedims\opts.nwindow\(npar*ndqn_max));
  	val action_memory = izeros(ndqn_max\npar);
  	val reward_memory = zeros(ndqn_max\npar);
  	val done_memory = zeros(ndqn_max\npar);
  	reward_plot = zeros(1, nsteps/printsteps0);

  	tic;
  	istep = ndqn_max;
  	for (i <- 0 until npar) old_lives(i) = envs(i).lives();
  	val epsilonvec0 = exp(- ln(opts.lambda) * (1 - row(0->npar)/npar));                // per-thread epsilons
  	var nupdates = 0;
  	
  	myLogger.info("Started Training");
  	while (istep <= opts.nsteps && !done) {
//    if (render): envs[0].render()
  		val lr = opts.lr_schedule(istep);                                          // Update the decayed learning rate
  		val epsilon = opts.eps_schedule(istep);                                    // Get an epsilon for the eps-greedy policy
  		val epsilonvec = epsilonvec0 * epsilon;

  		var i = 0;
  		val rr = math.log(u0 + (1-u0)*rn.nextDouble())/v0;
  		val xdqn = math.max(1, math.min(ndqn_max, 1 + math.floor(rr).toInt));
  		xhist(0, xdqn-1) += 1;
  		
  		if ((istep+xdqn) % targwin < ndqn_max && istep % targwin >= ndqn_max) t_estimator.update_from(q_estimator);        // Update the target estimator if needed    

  		while (i < xdqn && !done) {
  			times(0) = toc;
  			zstate ~ state - mean_state;
  			q_estimator.predict(zstate);                                           // get the next action probabilities etc from the policy
  			val (preds, _) = q_estimator.getOutputs2;
  			times(1) = toc;

  			val probs = (maxi(preds) == preds);
  			probs ~ probs / sum(probs);
  			if (i == xdqn-1 || ! opts.q_exact_policy) {                            // if score_exact dont epsilon-blend in environment 0
  			  probs(?,irange) = epsilonvec(0,irange) *@ rand_actions(?,irange) + (1-epsilonvec(0,irange)) *@ probs(?,irange);          
  			}
  			actions <-- multirnd(probs);                                           // Choose actions using the policy 
  			val (obs, rewards, dones) = parstepper(envs, VALID_ACTIONS(actions), obs0, rewards0, dones0);           // step through parallel envs
  			for (i <- 0 until npar) new_lives(i) = envs(i).lives();
  			times(2) = toc;

  			for (j <- 0 until npar) {                                              // process the observation
  				new_state(?,?,0->(nwindow-1),j) = state(?,?,1->nwindow,j);           // shift the image stack and add a new image
  				new_state(?,?,nwindow-1,j) = obs(j);         
  			}    
  			saved_frames(?,?,igame) = obs(0).reshapeView(envs(0).statedims\1);
  			saved_actions(0,igame) = actions(0);
  			saved_preds(?,igame) = preds(?,0);
  			saved_rewards(0,igame) = rewards(0);
  			saved_dones(0,igame) = dones(0);
  			saved_lives(0,igame) = envs(0).lives();
  			igame = (igame+1) % save_length;
  			
  			nupdates += 1;
  			val action_probs = probs(actions + irow(0->probs.ncols)*probs.nrows);
  			if (opts.nexact > 0) {
  				total_epochs += sum(dones(0->opts.nexact)).v.toInt;
  				block_reward += sum(rewards(0->opts.nexact)).v;
  				block_entropy -= sum(ln(action_probs(0->opts.nexact))).v/opts.nexact;
  			} else {
  				total_epochs += sum(dones).v.toInt;
  				block_reward += sum(rewards).v;
  				block_entropy -= sum(ln(action_probs)).v/action_probs.length;
  			}
  			min(rewards, envs(0).opts.limit_reward_incr(1), rewards)
  			max(rewards, envs(0).opts.limit_reward_incr(0), rewards)
  			
  			times(3) = toc;

  			if (envs(0).opts.endEpochAtReward) {
  				dones <-- (dones + (rewards != 0f) > 0f);
  			}
  			
  			if (envs(0).opts.endEpochAtDeath) {
  				dones <-- (dones + (new_lives < old_lives) > 0f);
  			}
  			old_lives <-- new_lives;

  			state_memory(?,?,?,(i*npar)->((i+1)*npar)) = state;
  			action_memory(i,?) = actions;
  			reward_memory(i,?) = rewards;
  			done_memory(i,?) = dones;
  			state <-- new_state;
  			times(4) = toc;
  			dtimes(0,0->4) = dtimes(0,0->4) + (times(0,1->5) - times(0,0->4));
  			while (paused || (pauseAt > 0 && istep + i >= pauseAt)) Thread.sleep(1000);
  			i += 1;
  		}
  		zstate ~ new_state - mean_state;
  		t_estimator.predict(zstate);
  		val (q_next, _) = t_estimator.getOutputs2; 
  		val probs = (maxi(q_next) == q_next);
  		probs ~ probs / sum(probs);
  	  if (! opts.q_exact_policy) {                            // if score_exact dont epsilon-blend
  				probs(?,irange) = epsilonvec(0,irange) *@ rand_actions(?,irange) + (1-epsilonvec(0,irange)) *@ probs(?,irange);          
  	  }
  		val v_next = q_next dot probs;
  		times(5) = toc;

  		reward_memory(xdqn-1,?) = reward_memory(xdqn-1,?) + (1f-done_memory(xdqn-1,?)) *@ opts.discount_factor *@ v_next ; // Propagate rewards from Q-values at non-final states.
  		for (i <- (xdqn-2) to 0 by -1) {
  			// Propagate rewards back in time, but not across epochs. 
  			reward_memory(i,?) = reward_memory(i,?) + (1f - done_memory(i,?)) *@ opts.discount_factor *@ reward_memory(i+1,?);
  		}

  		// Now compute gradients for the states/actions/rewards saved in the table.
  		for (i <- 0 until xdqn) {
  			new_state <-- state_memory(?,?,?,(i*npar)->((i+1)*npar));
  			zstate ~ new_state - mean_state;
  			q_estimator.gradient(zstate, action_memory(i,?), reward_memory(i,?), npar);
  			val (_, lv) = q_estimator.getOutputs2;
  			block_loss += sum(lv).v;                                // compute q-estimator gradient and return the loss 
  		}
  		times(6) = toc;

  		q_estimator.msprop(lr);                       // apply the gradient update
  		times(7) = toc;

  		dtimes(0,4->7) = dtimes(0,4->7) + (times(0,5->8) - times(0,4->7));
  		val t = toc;
  		if ((istep+xdqn) % printsteps0 < ndqn_max && istep % printsteps0 >= ndqn_max) {
  			total_reward += block_reward;
  			myLogger.info("Iter %5d, Time %4.1f, Loss %7.6f, Ent %5.4f, Epoch %d, Rew/Ep %5.4f, Cum Rew/Ep %5.4f" 
  					format((istep+xdqn)/printsteps0*printsteps0, t, block_loss/printsteps0/npar, block_entropy/printsteps0, 
  							total_epochs, block_reward/math.max(1,total_epochs-last_epochs), total_reward/math.max(1,total_epochs)));
  			reward_plot((istep+xdqn)/printsteps0-1) = block_reward/math.max(1,total_epochs-last_epochs);
  			last_epochs = total_epochs;
  			block_reward = 0f;
  			block_loss = 0f;
  			block_entropy = 0f;
  		}
  		istep += xdqn;
  	}
  	Mat.useGPUcache = GPUcacheState;
  	Mat.useCache = cacheState;
  }
}

object NDQNalgorithm {
  trait Opts extends Algorithm.Opts {
    
    logfile = "logNDQN.txt";
    tensorFormat = Net.TensorNCHW;
    
    var nsteps = 400000;                             // Number of steps to run (game actions per environment)
  	var ndqn = 5;                                    // Number of DQN steps per update
  	var ndqn_max = 0;
  	var target_window = 50;                          // Interval to update target estimator from q-estimator
  	var print_steps = 10000;                         // Number of steps between printouts
  	var init_moves = 4000;                           // Upper bound on random number of moves to take initially
  	var nwindow = 4;                                 // Sensing window = last n images in a state
  	
  	var discount_factor = 0.99f;                     // Reward discount factor
  	var q_exact_policy = false;                      // Compute Q values for the true policy vs. exploration policy (like DeepMind)
  	var nexact = 0;                                  // Score the true policy only (in envs 0->nexact)
  	var lambda = 10f;                                // Spread ratio of per-policy epsilons
  	
  	var lr_schedule:FMat = null;                     // Learning rate schedule
  	var eps_schedule:FMat = null;                    // Epsilon schedule
  }
  
  class Options extends Opts {}
}
