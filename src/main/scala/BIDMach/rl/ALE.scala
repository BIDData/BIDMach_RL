package BIDMach.rl;
import BIDMat.{IMat,FMat};
import BIDMat.MatFunctions._;
import java.util.Random;

class ALE extends edu.berkeley.bid.ALE {
    var dims:Array[Int] = null;
    var buffer:Array[Byte] = null;
    val rg = new Random();
    var frameskip = (2, 5);
    var mode = 0;           // 0 = Raw, 1 = Grayscale, 2 = RGB

    def getBufferData:Array[Byte] = {
	buffer = mode match {
	case 0 => getScreenData(buffer);
	case 1 => getScreenGrayscale(buffer);
	case 2 => getScreenRGB(buffer);
	}
	buffer;
    }	

    def copyObs(out0:FMat):FMat = {
	if (dims == null) {
	    dims = getScreenDims
	}
	val width = dims(0);
	val height = dims(1);
	val out = if (out0.asInstanceOf[AnyRef] == null) {
	    mode match {
	    case 0 => zeros(width \ height);
	    case 1 => zeros(width \ height);
	    case 2 => zeros(3 \ width \ height);
	    }
	} else {
	    out0;
	}
	getBufferData;
	val odata = out.data;
	val len = if (mode == 1) width*height*3 else width*height;
	var i = 0;
	while (i < len) {
	    odata(i) = (buffer(i)) & 0xff;
	    i += 1;
	}
	out;
    }

    def copyObs():FMat = copyObs(null);

    def step(action:Int, out0:FMat):(FMat, Float, Boolean) = {
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
	val out = copyObs(out0);
	val done = game_over();
	(out, reward, done)
    };

    def step(action:Int):(FMat, Float, Boolean) = step(action, null);

    def step2(action:Int):(Array[Byte], Float, Boolean) = {
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
	val out = getBufferData;
	val done = game_over();
	(out, reward, done)
    };
  
    def reset(out0:FMat):FMat = {
	reset_game();
	copyObs(out0);
    };
  
    def reset():FMat = reset(null);

    override def loadROM(s:String):Int = {
	val status = super.loadROM(s);
	dims = null;
	buffer = null;
	status;
    }
}

object ALE {
    val rg = new Random();

    def stepAll(envs:Array[ALE],  actions:IMat, obs0:Array[FMat], rewards0:FMat, dones0:FMat):(Array[FMat], FMat, FMat) = {
	val npar = envs.length;
	val obs = if (obs0.asInstanceOf[AnyRef] == null) new Array[FMat](npar) else obs0;
	val rewards = if (rewards0.asInstanceOf[AnyRef] == null) zeros(1, npar) else rewards0;
	val dones = if (dones0.asInstanceOf[AnyRef] == null) zeros(1, npar) else dones0;
	(0 until npar).par.foreach((i) => {
		val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
		rewards(i) = (0 until nsteps).map((j) => envs(i).act(actions(i))).sum;
		obs(i) = envs(i).copyObs(obs(i));
		dones(i) = if (envs(i).game_over()) 1f else 0f;
		if (dones(i) == 1f) envs(i).reset_game();
	    })
	(obs, rewards, dones)
    };
	
    def stepAll(envs:Array[ALE], actions:IMat):(Array[FMat], FMat, FMat) = stepAll(envs, actions, null, null, null);

    def stepAll2(envs:Array[ALE], actions:IMat, obs0:Array[Array[Byte]], rewards0:FMat, dones0:FMat):(Array[Array[Byte]], FMat, FMat) = {
	val npar = envs.length;
	val obs = if (obs0.asInstanceOf[AnyRef] == null) new Array[Array[Byte]](npar) else obs0;
	val rewards = if (rewards0.asInstanceOf[AnyRef] == null) zeros(1, npar) else rewards0;
	val dones = if (dones0.asInstanceOf[AnyRef] == null) zeros(1, npar) else dones0;
	(0 until npar).par.foreach((i) => {
		val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
		rewards(i) = (0 until nsteps).map((j) => envs(i).act(actions(i))).sum;
		obs(i) = envs(i).getBufferData;
		dones(i) = if (envs(i).game_over()) 1f else 0f;
		if (dones(i) == 1f) envs(i).reset_game();
	    })
	(obs, rewards, dones)
    };
	
    def stepAll2(envs:Array[ALE], actions:IMat):(Array[Array[Byte]], FMat, FMat) = stepAll2(envs, actions, null, null, null);
}


