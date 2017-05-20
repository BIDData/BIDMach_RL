package BIDMach.rl.environments;

import BIDMat.{IMat,FMat};
import BIDMat.MatFunctions._;
import java.util.Random;

class ALE extends edu.berkeley.bid.ALE {
	var dims:Array[Int] = null;
  var buffer:Array[Byte] = null;
  var buffer2:Array[Byte] = null;
  val rg = new Random();
  var frameskip = (2, 5);
  var mode = 0;           // 0 = Raw, 1 = Grayscale, 2 = RGB, 3 = binarize
  var pool = true;      
  var shrink = true;
  var background = 0;
  var xoff = 0;
  var yoff = 17;
  var width = 80;
  var height = 80;

  def getBufferData(buffer:Array[Byte]):Array[Byte] = {
    val buffer0 = mode match {
    case 0 => getScreenData(buffer);
    case 1 => if (pool) getScreenRGB(buffer) else getScreenGrayscale(buffer);
    case 2 => getScreenRGB(buffer);
    case 3 => getScreenData(buffer);
    }
    buffer0;
  };

  def copyObs(out0:FMat):FMat = {
		if (dims == null) {
			dims = getScreenDims
		}
		val inwidth = dims(0);
		val inheight = dims(1);
		val out = if (out0.asInstanceOf[AnyRef] == null) {
			if (mode == 2) {
				zeros(3 \ width \ height);
			} else {
				zeros(width, height);	  
			}
		} else {
			out0;
		}
		val odata = out.data;
		var i = 0;
		if (pool) {
			if (shrink) {
				out.clear;
				mode match {
				  case 0 => throw new RuntimeException("Cant do pooling and shrinking on a native image");
				  case 1 => {                                // Pool and shrink a grayscale image
				  	val cc = 1f/3/255/4;
				  	while (i < height*2) {
				  		val irow = (i + yoff) * inwidth * 3;
				  		val irow2 = (i >> 1) * width;
				  		var j = 0;
				  		while (j < width*2) {
				  			val ii = irow + (j + xoff)*3;
				  			val jj = irow2 + (j >> 1);
				  			val r = math.max(buffer(ii) & 0xff, buffer2(ii) & 0xff);
				  			val g = math.max(buffer(ii+1) & 0xff, buffer2(ii+1) & 0xff);
				  			val b = math.max(buffer(ii+2) & 0xff, buffer2(ii+2) & 0xff);
				  			odata(jj) += (r+g+b)*cc;
				  			j += 1;					
				  		}
				  		i += 1;
				  	}		
				  }
				  case 2 => {                                // Pool and shrink a color image
				  	val cc = 1f/255/4;
				  	val len = width * height;
				  	if (i < height*2) {
				  		val irow = (i + yoff) * inwidth * 3;
				  		val irow2 = (i >> 1) * width;
				  		var j = 0;
				  		while (j < width*2) {
				  			val ii = irow + (j + xoff)*3;
				  			val jj = irow2 + (j >> 1);
				  			val r = math.max(buffer(ii) & 0xff, buffer2(ii) & 0xff);
				  			val g = math.max(buffer(ii+1) & 0xff, buffer2(ii+1) & 0xff);
				  			val b = math.max(buffer(ii+2) & 0xff, buffer2(ii+2) & 0xff);
				  			odata(jj) += r*cc;
				  			odata(jj + len) += g*cc;
				  			odata(jj + len + len) += b*cc;
				  			j += 1;					
				  		}
				  		i += 1;
				  	}		
				  }	
				  case 3 => {                                // Pool and shrink a binarized image
				  	val cc = 1f/4;
				  	while (i < height*2) {
				  		val irow = (i + yoff) * inwidth;
				  		val irow2 = (i >> 1) * width;
				  		var j = 0;
				  		while (j < width*2) {
				  			val ii = irow + j + xoff;
				  			val jj = irow2 + (j >> 1);
				  			val r = if ((buffer(ii) & 0xff) != background || (buffer2(ii) & 0xff) != background) cc else 0f;
				  			odata(jj) += r;
				  			j += 1;					
				  		}
				  		i += 1;
				  	}	
				  }
				}
		  } else {                                       // Pooling without shrinking
		    mode match {
		      case 0 => throw new RuntimeException("Cant do pooling on a native image");
		      case 1 => {                                // Pool a grayscale image
		      	val cc = 1f/3/255;
		      	while (i < height) {
		      		val irow = (i + yoff) * inwidth * 3;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + (j + xoff)*3;
		      			val jj = irow2 + j ;
		      			val r = math.max(buffer(ii) & 0xff, buffer2(ii) & 0xff);
		      			val g = math.max(buffer(ii+1) & 0xff, buffer2(ii+1) & 0xff);
		      			val b = math.max(buffer(ii+2) & 0xff, buffer2(ii+2) & 0xff);
		      			odata(jj) += (r+g+b)*cc;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}	
		      }
		      case 2 => {                                // Pool a color image
		        val cc = 1f/255;
		        val len = width * height;
		      	while (i < height) {
		      		val irow = (i + yoff) * inwidth * 3;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + (j + xoff)*3;
		      			val jj = irow2 + j;
		      			val r = math.max(buffer(ii) & 0xff, buffer2(ii) & 0xff);
		      			val g = math.max(buffer(ii+1) & 0xff, buffer2(ii+1) & 0xff);
		      			val b = math.max(buffer(ii+2) & 0xff, buffer2(ii+2) & 0xff);
		      			odata(jj) += r*cc;
		      			odata(jj + len) += g*cc;
		      			odata(jj + len + len) += b*cc;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}	
		      }
		      case 3 => {                                // Pool a binarized image
		        while (i < height) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + j;
		      			odata(jj) = if ((buffer(ii) & 0xff) != background || (buffer2(ii) & 0xff) != background) 1f else 0f;
		      			j += 1;					
		      		}
		      		i += 1;
		        }
		      }
		    }
		  }
		} else {                                         // Dont pool
		  if (shrink) {                                  // Do shrink
		    mode match {
		      case 0 => {                                // Shrink a raw image by downsampling
		      	while (i < height*2) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = (i >> 1) * width;
		      		var j = 0;
		      		while (j < width*2) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + (j  >> 1);
		      			odata(jj) = (buffer(ii) & 0xff);
		      			j += 1;					
		      		}
		      		i += 1;
		      	}		        
		      }
		      case 1 => {                                // Shrink a grayscale imag
		        out.clear;
		      	val cc = 1f/255/4;
		      	while (i < height*2) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = (i >> 1) * width;
		      		var j = 0;
		      		while (j < width*2) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + (j >> 1);
		      			odata(jj) += (buffer(ii) & 0xff) * cc;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}
		      }
		      case 2 => {                                // Shrink a color image
		      	out.clear;
		      	val cc = 1f/255/4;
		      	val len = width * height;
		      	while (i < height*2) {
		      		val irow = (i + yoff) * inwidth * 3;
		      		val irow2 = (i >> 1) * width;
		      		var j = 0;
		      		while (j < width*2) {
		      			val ii = irow + (j + xoff)*3;
		      			val jj = irow2 + (j >> 1);
		      			odata(jj) += (buffer(ii) & 0xff) * cc;
		      			odata(jj + len) += (buffer(ii+1) & 0xff) * cc;
		      			odata(jj + len + len) += (buffer(ii+2) & 0xff) * cc;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}
		      }
		      case 3 => {                                // Shrink a binarized image
		      	out.clear;
		      	val cc = 1f/4;
		      	while (i < height*2) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = (i >> 1) * width;
		      		var j = 0;
		      		while (j < width*2) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + (j >> 1);
		      			val r = if ((buffer(ii) & 0xff) != background) cc else 0f;
		      			odata(jj) += r;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}		
		      }
		    }
		  } else {                                           // No pooling, no shrinking
		    mode match {
		      case 0 => {                                    // Copy a raw image
		      	while (i < height) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + j;
		      			odata(jj) = buffer(ii) & 0xff;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}			        
		      }
		      case 1 => {                                    // Copy a grayscale image
		        val cc = 1f/255;
		        while (i < height) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + j;
		      			odata(jj) = (buffer(ii) & 0xff) * cc;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}		        
		      }
		      case 2 => {                                    // Copy an RGB image
		        val cc = 1f/255;
		        val len = width * height;
		        while (i < height) {
		      		val irow = (i + yoff) * inwidth * 3;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + (j + xoff)*3;
		      			val jj = irow2 + j;
		      			odata(jj) = (buffer(ii) & 0xff) * cc;
		      			odata(jj+len) = (buffer(ii+1) & 0xff) * cc;
		      			odata(jj+len+len) = (buffer(ii+2) & 0xff) * cc;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}
		      }
		      case 3 => {                                    // Copy a binarized image
		         while (i < height) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + j;
		      			odata(jj) = if ((buffer(ii) & 0xff) != background) 1f else 0f;
		      			j += 1;					
		      		}
		      		i += 1;
		      	}		 		        
		      }
		    }
		  }
		}
		out;
  };
  
  

  def copyObs():FMat = copyObs(null);

  def step(action:Int, out0:FMat):(FMat, Float, Boolean) = {
  	val nsteps = frameskip._1 + rg.nextInt(frameskip._2 - frameskip._1 + 1);
  	if (dims == null) {
  		dims = getScreenDims
  	}
  	var reward = 0f;
  	var i = 0;
  	while (i < nsteps) {
  		reward += act(action);
  		if (pool && i == nsteps-2) buffer2 = getBufferData(buffer2);
  		i += 1;
  	} 
  	buffer = getBufferData(buffer);
  	val out = copyObs(out0);
  	val done = game_over();
  	(out, reward, done)
  };

  def step(action:Int):(FMat, Float, Boolean) = step(action, null);

  def step2(action:Int):(Array[Byte], Float, Boolean) = {
  	val nsteps = frameskip._1 + rg.nextInt(frameskip._2 - frameskip._1 + 1);
  	if (dims == null) {
  		dims = getScreenDims
  	}
  	var reward = 0f;
  	var i = 0;
  	while (i < nsteps) {
  		reward += act(action);
  		i += 1;
  	} 
  	val out = getBufferData(buffer);
  	val done = game_over();
  	(out, reward, done)
  };

  def reset(out0:FMat):FMat = {
  		reset_game();
  		buffer = getBufferData(buffer);
  		if (pool) buffer2 = getBufferData(buffer2);
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
			var reward = 0f;
			for (i <- 0 until nsteps) {
			  reward += envs(i).act(actions(i));
			  if (envs(i).pool & i == nsteps-2) envs(i).buffer2 = envs(i).getBufferData(envs(i).buffer2);
			}
			rewards(i) = rewards;
			envs(i).buffer = envs(i).getBufferData(envs(i).buffer);
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
			obs(i) = envs(i).getBufferData(envs(i).buffer);
			dones(i) = if (envs(i).game_over()) 1f else 0f;
			if (dones(i) == 1f) envs(i).reset_game();
		})
		(obs, rewards, dones)
	};

	def stepAll2(envs:Array[ALE], actions:IMat):(Array[Array[Byte]], FMat, FMat) = stepAll2(envs, actions, null, null, null);
}


