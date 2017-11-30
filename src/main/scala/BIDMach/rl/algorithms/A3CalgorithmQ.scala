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

class A3CalgorithmQ(
		val envs:Array[Environment], 
		val parstepper:(Array[Environment], IMat, Array[FMat], FMat, FMat) => (Array[FMat], FMat, FMat),
		val buildEstimator:(Estimator.Opts) => Estimator,
		val opts:A3CalgorithmQ.Opts = new A3CalgorithmQ.Options
		) extends Algorithm(opts) {
  
	val npar = envs.length;                            // Number of parallel environments 
	
	var VALID_ACTIONS:IMat = null;
	var nactions = 0;
	var total_steps = 0;
	var block_reward = 0f;
	var total_reward = 0f;
	var block_entropy = 0f;
	var block_loss = 0f;
	var block_count = 0;
	var total_epochs = 0;
	var total_time = 0f;
	var igame = 0;
	var state:FMat = null;
	var obs0:FMat = null;
	val rn = new java.util.Random;
	
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
	  saved_rewards = zeros(1, save_length);
	  saved_dones = zeros(1, save_length);
	  saved_preds = zeros(nactions\save_length);
	  
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
  	val GPUcacheState = Mat.useGPUcache;
  	val cacheState = Mat.useCache;
  	Mat.useGPUcache = true;
  	Mat.useCache = false;
    
// Create estimators
  	estimator = buildEstimator(opts.asInstanceOf[Estimator.Opts]);
  	estimator.predict(state);    //	Initialize them by making predictions
  	  	
  	val times = zeros(1,8);
  	dtimes.clear;
  	val ractions = int(rand(1, npar) * nactions);
  	val (obs0, rewards0, dones0) = parstepper(envs, VALID_ACTIONS(ractions), null, null, null);           // step through parallel envs
 
  	var actions = izeros(1,npar);
  	var action_probs:FMat = null;
  	val rand_actions = ones(nactions, npar) * (1f/nactions); 
  	val printsteps0 = opts.print_steps / ndqn * ndqn; 
  	  	
  	val state_memory = zeros(envs(0).statedims\opts.nwindow\(npar*ndqn));
  	val action_memory = izeros(ndqn\npar);
  	val reward_memory = zeros(ndqn\npar);
  	val done_memory = zeros(ndqn\npar);
  	reward_plot = zeros(1, nsteps/printsteps0);

  	tic;
  	var istep = ndqn;
  	myLogger.info("Started Training");
  	while (istep <= opts.nsteps && !done) {
//    if (render): envs[0].render()
  		val lr = learning_rates(istep);                                // update the decayed learning rate
  		val temp = temperatures(istep);                                // get an epsilon for the eps-greedy policy
  		estimator.setConsts3(1/temp, opts.entropy_weight, opts.policygrad_weight);

  		var i = 0;
  		while (i < ndqn && !done) {
  			times(0) = toc;
  			estimator.predict(state); // get the next action probabilities etc from the policy
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
  			saved_rewards(0,igame) = rewards(0);
  			saved_preds(0,igame) = preds(0);
  			saved_preds(?,igame) = preds(?,0);
  			igame = (igame+1) % save_length;
  			
  			total_epochs += sum(dones).v.toInt;
  			block_reward += sum(rewards).v;

  			times(3) = toc;

  			if (envs(0).opts.endEpochAtReward) {
  				dones <-- (dones + (rewards != 0f) > 0f);
  			}

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
  		estimator.predict(new_state);
  		val (q_next, q_prob, _, _) = estimator.getOutputs4; 
  		val v_next = q_next dot q_prob;
  		times(5) = toc;

  		reward_memory(ndqn-1,?) = reward_memory(ndqn-1,?) + (1f-done_memory(ndqn-1,?)) *@ v_next; // add actual and predicted reward, unless at end of epoch
  		for (i <- (ndqn-2) to 0 by -1) {
  			// Propagate rewards back in time, not crossing epoch boundary
  			reward_memory(i,?) = reward_memory(i,?) + (1f - done_memory(i,?)) *@ reward_memory(i+1,?) *@ opts.discount_factor;
  		}

  		// Now compute gradients for the states/actions/rewards saved in the table.
  		for (i <- 0 until ndqn) {
  			new_state <-- state_memory(?,?,?,(i*npar)->((i+1)*npar));
  			estimator.gradient(new_state, action_memory(i,?), reward_memory(i,?), npar);
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
  			myLogger.info("I %5d, T %4.1f, L %7.6f, Ent %5.4f, E %d, R/E %5.4f, CR/E %5.4f" 
  					format(istep, t, block_loss/printsteps0/npar, block_entropy/printsteps0/npar, 
  							total_epochs, block_reward/math.max(1,total_epochs-last_epochs), total_reward/math.max(1,total_epochs)));
  			reward_plot(istep/printsteps0-1) = block_reward/math.max(1,total_epochs-last_epochs);
  			last_epochs = total_epochs;
  			block_reward = 0f;
  			block_loss = 0f;
  			block_entropy = 0f;
  		}
  		istep += ndqn;
  	}
  	Mat.useGPUcache = GPUcacheState;
  	Mat.useCache = cacheState;
  }
}

object A3CalgorithmQ {
  trait Opts extends Algorithm.Opts {
    
  	logfile = "logA3CQ.txt";
    tensorFormat = Net.TensorNCHW;
    
    var nsteps = 400000;                             // Number of steps to run (game actions per environment)
  	var ndqn = 5;                                    // Number of DQN steps per update
  	var print_steps = 10000;                         // Number of steps between printouts
  	var init_moves = 4000;                           // Upper bound on random number of moves to take initially
  	var nwindow = 4;                                 // Sensing window = last n images in a state
  	
  	var discount_factor = 0.99f;                     // Reward discount factor
  	var policygrad_weight = 0.3f;                    // Weight of policy gradient compared to regression loss
  	var entropy_weight = 1e-4f;                      // Entropy regularization weight
  	
  	var lr_schedule = (0f \ 3e-6f on 1f \ 3e-6f);    // Learning rate schedule
  	var temp_schedule = (0f \ 1f on 1f \ 1f);        // Temperature schedule
  }
  
  class Options extends Opts {}
}