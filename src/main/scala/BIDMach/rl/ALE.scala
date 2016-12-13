package BIDMach.rl;
import BIDMat.{IMat,FMat,FND};
import BIDMat.MatFunctions._;
import java.util.Random;

class ALE extends edu.berkeley.bid.ALE {
	var dims:Array[Int] = null;
  var buffer:Array[Byte] = null;
  val rg = new Random();
  var frameskip = (2, 5);

  def copyObs(out0:FND):FND = {
		if (dims == null) {
			dims = getScreenDims
		}
		buffer = getScreenRGB(buffer);
		val width = dims(0);
		val height = dims(1);
		val len = width*height;
		val out = if (out0.asInstanceOf[AnyRef] == null) FND(3, width, height) else out0;
		val odata = out.data;
		var i = 0;
		while (i < len*3) {
			odata(i) = (buffer(i)) & 0xff;
			i += 1;
		}
		out;
  }
  
  def copyObs():FND = copyObs(null);

  def step(action:Int, out0:FND):(FND, Float, Boolean) = {
  	val nsteps = frameskip._1 + rg.nextInt(frameskip._2 - frameskip._1 + 1);
  	var reward = 0f;
  	var i = 0;
  	while (i < nsteps) {
  		reward += act(action);
  		i += 1;
  	} 
  	if (dims == null) {
			dims = getScreenDims
		}
 		val width = dims(0);
		val height = dims(1);
		val len = width*height; 	
  	val out = copyObs(out0);
  	val done = game_over();
  	(out, reward, done)
  };
  
  def step(action:Int):(FND, Float, Boolean) = step(action, null);

  def reset(out0:FND):FND = {
		reset_game();
		copyObs(out0);
  };
  
  def reset():FND = reset(null);

  override def loadROM(s:String):Int = {
  		val status = super.loadROM(s);
  		dims = null;
  		buffer = null;
  		status;
  }
}

object ALE {
	val rg = new Random();

	def stepAll(envs:Array[ALE],  actions:IMat, obs0:Array[FND], rewards0:FMat, dones0:FMat):(Array[FND], FMat, FMat) = {
		val npar = envs.length;
		val obs = if (obs0.asInstanceOf[AnyRef] == null) new Array[FND](npar) else obs0;
		val rewards = if (rewards0.asInstanceOf[AnyRef] == null) zeros(1, npar) else rewards0;
		val dones = if (dones0.asInstanceOf[AnyRef] == null) zeros(1, npar) else dones0;
		(0 until npar).par.foreach((i) => {
			val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
			rewards(i) = (0 until nsteps).map((j) => envs(i).act(actions(i))).sum;
			obs(i) = envs(i).copyObs(obs(i));
			dones(i) = if (envs(i).game_over()) 1f else 0f;
			if (dones(i) == 1f) envs(i).reset_game()
		})
		(obs, rewards, dones)
	};
	
	def stepAll(envs:Array[ALE], actions:IMat):(Array[FND], FMat, FMat) = stepAll(envs, actions, null, null, null);

	def stepAll2(envs:Array[ALE], actions:IMat, obs0:Array[Array[Byte]], rewards0:FMat, dones0:FMat):(Array[Array[Byte]], FMat, FMat) = {
		val npar = envs.length;
		val obs = if (obs0.asInstanceOf[AnyRef] == null) new Array[Array[Byte]](npar) else obs0;
		val rewards = if (rewards0.asInstanceOf[AnyRef] == null) zeros(1, npar) else rewards0;
		val dones = if (dones0.asInstanceOf[AnyRef] == null) zeros(1, npar) else dones0;
		(0 until npar).par.foreach((i) => {
			val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
			rewards(i) = (0 until nsteps).map((j) => envs(i).act(actions(i))).sum;
			obs(i) = envs(i).getScreenData(null);
			dones(i) = if (envs(i).game_over()) 1f else 0f;
			if (dones(i) == 1f) envs(i).reset_game()
		})
		(obs, rewards, dones)
	};
	
	def stepAll2(envs:Array[ALE], actions:IMat):(Array[Array[Byte]], FMat, FMat) = stepAll2(envs, actions, null, null, null);

}


