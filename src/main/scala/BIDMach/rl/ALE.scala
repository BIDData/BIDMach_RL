package BIDMach.rl;
import BIDMat.{IMat,FMat,FND};
import BIDMat.MatFunctions._;
import java.util.Random;

class ALE extends edu.berkeley.bid.ALE {
    var dims:Array[Int] = null;
    var buffer:Array[Byte] = null;
    val rg = new Random();
    var frameskip = (2, 5);

    def copyObs():FND = {
	if (dims == null) {
	    dims = getScreenDims
	}
	buffer = getScreenRGB(buffer);
	val width = dims(0);
	val height = dims(1);
	val len = width*height;
	val out = FND(3, width, height);
	val odata = out.data;
	var i = 0;
	while (i < len*3) {
	    odata(i) = (buffer(i)) & 0xff;
	    i += 1;
	}
	out;
    }
      
    def step(action:Int):(FND, Float, Boolean) = {
        val nsteps = frameskip._1 + rg.nextInt(frameskip._2 - frameskip._1 + 1);
	var reward = 0f;
	var i = 0;
	while (i < nsteps) {
	    reward += act(action);
	    i += 1;
	} 
	val out = copyObs();
	val done = game_over();
	(out, reward, done)
    };

    def reset():FND = {
	reset_game();
	copyObs();
    };

    override def loadROM(s:String):Int = {
	val status = super.loadROM(s);
	dims = null;
	buffer = null;
	status;
    }
}

object ALE {
    val rg = new Random();

    def stepAll(envs:Array[ALE],  actions:IMat):(Array[FND], FMat, FMat) = {
	val npar = envs.length;
	val rewards = zeros(1, npar);
	val dones = zeros(1, npar);
	val obs = new Array[FND](npar);
	(0 until npar).par.foreach((i) => {
		val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
	        rewards(i) = (0 until nsteps).map((j) => envs(i).act(actions(i))).sum;
		obs(i) = envs(i).copyObs();
		dones(i) = if (envs(i).game_over()) 1f else 0f;
		if (dones(i) == 1f) envs(i).reset_game()
	    })
	(obs, rewards, dones)
    };

    def stepAll2(envs:Array[ALE],  actions:IMat):(Array[Array[Byte]], FMat, FMat) = {
	val npar = envs.length;
	val rewards = zeros(1, npar);
	val dones = zeros(1, npar);
	val obs = new Array[Array[Byte]](npar);
	(0 until npar).par.foreach((i) => {
		val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
	        rewards(i) = (0 until nsteps).map((j) => envs(i).act(actions(i))).sum;
		obs(i) = envs(i).getScreenData(null);
		dones(i) = if (envs(i).game_over()) 1f else 0f;
		if (dones(i) == 1f) envs(i).reset_game()
	    })
	(obs, rewards, dones)
    };

}


