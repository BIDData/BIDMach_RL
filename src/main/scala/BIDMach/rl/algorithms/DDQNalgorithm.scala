package BIDMach.rl.algorithms;

/*
 * Double DQN and Double Q-Learning.
 * 
 * Both methods use the Double Q-Learning target value based on two estimators. 
 * DDQN uses a lagged target estimator as the second estimator. 
 * DQL uses two estimators that alternate roles.
 * 
 * If target_window > 0, then DDQN is used with target estimator updated every target_window steps. 
 * If target_window == 0, then DQL is used. 
 */

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

class DDQNalgorithm(
		val envs:Array[Environment], 
		val parstepper:(Array[Environment], IMat, Array[FMat], FMat, FMat) => (Array[FMat], FMat, FMat),
		val buildEstimator:(Estimator.Opts) => Estimator,
		val opts:DDQNalgorithm.Opts = new DDQNalgorithm.Options
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
	var actionColOffsets:IMat = null;
	val rn = new java.util.Random;
	
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
	  actionColOffsets = irow(0->npar) * nactions;
	  
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
    val old_lives = zeros(1, npar);
    val new_lives  = zeros(1, npar);
    val irange = irow(opts.nexact->npar);
    val targwin = opts.target_window;
    
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
  	val printsteps0 = opts.print_steps;
  	reward_plot = zeros(1, nsteps/printsteps0);

  	tic;
  	istep = 1;
  	for (i <- 0 until npar) old_lives(i) = envs(i).lives();
  	val epsilonvec0 = exp(- ln(opts.lambda) * (1 - row(0->npar)/npar));        // per-thread epsilons
  	
  	myLogger.info("Started Training");
  	while (istep < opts.nsteps && !done) {
  		val lr = opts.lr_schedule(istep);                                        // Update the decayed learning rate
  		val epsilon = opts.eps_schedule(istep);                                  // Get an epsilon for the eps-greedy policy
  		val epsilonvec = epsilonvec0 * epsilon;

  		if (targwin > 0 && (istep % targwin == 0)) t_estimator.update_from(q_estimator); // Update the target estimator if needed    

  		times(0) = toc;
  		zstate ~ state - mean_state;
  		q_estimator.predict(zstate);                                             // get the next action probabilities etc from the policy
  		val (q_next0, _) = q_estimator.getOutputs2;
  		times(1) = toc;
  		
  		val probs0 = (q_next0 == maxi(q_next0));                                 // Get the probabilities for the actions
  		probs0 ~ probs0 / sum(probs0);
  		probs0(?,irange) = epsilonvec(0,irange) *@ rand_actions(?,irange) + (1-epsilonvec(0,irange)) *@ probs0(?,irange);            		
  		actions <-- multirnd(probs0);                                            // Choose actions using the policy 
  		val (obs, rewards, dones) = parstepper(envs, VALID_ACTIONS(actions), obs0, rewards0, dones0);           // step through parallel envs
  		for (i <- 0 until npar) {new_lives(i) = envs(i).lives();}
  		times(2) = toc;

  		for (j <- 0 until npar) {                                                // Create the new state
  			new_state(?,?,0->(nwindow-1),j) = state(?,?,1->nwindow,j);             // shift the image stack and add a new image
  			new_state(?,?,nwindow-1,j) = obs(j);         
  		}    
  		saved_frames(?,?,igame) = obs(0).reshapeView(envs(0).statedims\1);       // Save diagnostic info for the first thread
  		saved_actions(0,igame) = actions(0);
  		saved_preds(?,igame) = q_next0(?,0);
  		saved_rewards(0,igame) = rewards(0);
  		saved_dones(0,igame) = dones(0);
  		saved_lives(0,igame) = envs(0).lives();
  		igame = (igame+1) % save_length;

  		val action_probs = probs0(actions + actionColOffsets);                   // Update performance stats for this step
  		if (opts.nexact > 0) {
  			total_epochs += sum(dones(0->opts.nexact)).v.toInt;
  			block_reward += sum(rewards(0->opts.nexact)).v;
  			block_entropy -= sum(ln(action_probs(opts.nexact->npar))).v/(npar-opts.nexact);
  		} else {
  			total_epochs += sum(dones).v.toInt;
  			block_reward += sum(rewards).v;
  			block_entropy -= sum(ln(action_probs)).v/action_probs.length;
  		}
  		min(rewards, envs(0).opts.limit_reward_incr(1), rewards)
  		max(rewards, envs(0).opts.limit_reward_incr(0), rewards)
  		times(3) = toc;

  		if (envs(0).opts.endEpochAtReward) {                                     // Close reward epoch at non-reward for some games
  			dones <-- (dones + (rewards != 0f) > 0f);
  		}

  		if (envs(0).opts.endEpochAtDeath) {                                      // Close reward epoch when life is lost
  			dones <-- (dones + (new_lives < old_lives) > 0f);
  		}
  		old_lives <-- new_lives;
  		times(4) = toc;
 
  		zstate ~ new_state - mean_state;                                         // Compute the Double-Q-Learning target
  		q_estimator.predict(zstate);
  		t_estimator.predict(zstate);
  		val (q_next1, _) = q_estimator.getOutputs2; 
  		val (t_next1, _) = t_estimator.getOutputs2;
  	  val (_, best_actions) = maxi2(q_next1);  	  
  	  val target = rewards + (1f - dones) *@ opts.discount_factor *@ t_next1(best_actions + actionColOffsets);
  		times(5) = toc;

  		zstate ~ state - mean_state;
  		q_estimator.gradient(zstate, actions, target, npar);                     // Compute q-estimator gradient and return the loss
  		val (_, lv) = q_estimator.getOutputs2;
  		block_loss += sum(lv).v;                                                 
  		times(6) = toc;
  		
  		while (paused || (pauseAt > 0 && istep >= pauseAt)) Thread.sleep(1000);

  		if (istep % opts.updateEvery == 0) {                                     // Update estimators with accumulated gradient
  			q_estimator.msprop(lr);                                                // Always apply the gradient update to q_estimator
  			if (targwin == 0) {                                                    // If we're doing Double Q-Learning, flip the estimators. 
  				val tmp_estimator = q_estimator;
  				q_estimator = t_estimator;
  				t_estimator = tmp_estimator;
  			}
  		}
  		state <-- new_state;                                                     // Move to the new state
  		times(7) = toc;

  		dtimes ~ dtimes + (times(0,1->8) - times(0,0->7));
  		val t = toc;
  		if (istep % printsteps0 == 0 ) {
  			total_reward += block_reward;
  			myLogger.info("Iter %5d, Time %4.1f, Loss %7.6f, Ent %5.4f, Epoch %d, Rew/Ep %5.4f, Cum Rew/Ep %5.4f" 
  					format(istep/printsteps0*printsteps0, t, block_loss/printsteps0/npar, block_entropy/printsteps0, 
  							total_epochs, block_reward/math.max(1,total_epochs-last_epochs), total_reward/math.max(1,total_epochs)));
  			reward_plot(istep/printsteps0-1) = block_reward/math.max(1,total_epochs-last_epochs);
  			last_epochs = total_epochs;
  			block_reward = 0f;
  			block_loss = 0f;
  			block_entropy = 0f;
  		}
  		istep += 1;
  	}
  	Mat.useGPUcache = GPUcacheState;
  	Mat.useCache = cacheState;
  }
}

object DDQNalgorithm {
  trait Opts extends Algorithm.Opts {
    
    logfile = "logDDQN.txt";
    tensorFormat = Net.TensorNCHW;
    
    var nsteps = 400000;                             // Number of steps to run (game actions per environment)
  	var updateEvery = 10;                             // Number of steps to update
  	var print_steps = 10000;                         // Number of steps between printouts
  	var init_moves = 4000;                           // Upper bound on random number of moves to take initially
  	var nwindow = 4;                                 // Sensing window = last n images in a state
  	var target_window = 50;                          // Interval to update target estimator from q-estimator. If 0, run Double QL instead.   	
  	var discount_factor = 0.99f;                     // Reward discount factor
  	var q_exact_policy = false;                      // Compute Q values for the true policy vs. exploration policy (like DeepMind)
  	var nexact = 0;                                  // Score the true policy only (in envs 0->nexact)
  	var lambda = 10f;                                // Spread ratio of per-policy epsilons
  	
  	var lr_schedule:FMat = null;                     // Learning rate schedule
  	var eps_schedule:FMat = null;                    // Epsilon schedule
  }
  
  class Options extends Opts {}
}
