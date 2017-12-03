package BIDMach.rl.environments;

import BIDMat.{IMat,FMat};
import BIDMat.MatFunctions._;
import java.util.Random;

class ALE extends edu.berkeley.bid.ALE {
	var dims:Array[Int] = null;
  var buffer:Array[Byte] = null;
  var buffer2:Array[Byte] = null;
  var colormap:Array[Float] = null
  val rg = new Random();
  var frameskip = (2, 5);
  var mode = 1;           // 0 = Raw, 1 = Grayscale, 2 = RGB, 3 = binarize, 4 = colormap
  var pool = true;      
  var shrink = true;
  var fire_to_start = false;
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
    case 4 => getScreenData(buffer);
    }
    buffer0;
  };
  
  def cinterp(x:Int, y:Int):Float = {
    (math.atan2(y, x) / Math.PI * 2).toFloat;
  }
  
  def cmap(r:Int, g:Int, b:Int):Float = {
    if (r==0 && g==0 & b==0) {
      0f
    } else {
      val rg = math.abs(r-g);
      val gb = math.abs(g-b);
      val rb = math.abs(r-b);
      if (rg > gb) {
        if (rg > rb) {
          0.1f + cinterp(r, g)*0.3f;
        } else {
        	0.7f + cinterp(b, r)*0.3f;
        }
      } else {
        if (gb > rb) {
          0.4f + cinterp(g, b)*0.3f;
        } else {
        	0.7f + cinterp(b, r)*0.3f;
        }
      }
    }
  }
  
  // The Standard Grayscale Palette gives very low constrast on many games. This map converts color angle into the output value.
  
  def createColormap:Array[Float] = {
    colormap = new Array[Float](256);
    val palette = getScreenPaletteRGB(null);
    for (i <- 0 until 256) {
      val r = palette(i*3) & 0xff;
      val g = palette(i*3+1) & 0xff;
      val b = palette(i*3+2) & 0xff;
      colormap(i) = cmap(r,g,b);
    }
    colormap;
  }

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
				  case 4 => {                                // Pool and shrink a colormapped image
				  	val cc = 1f/4;
				  	if (colormap.asInstanceOf[AnyRef] == null) colormap = createColormap;
				  	while (i < height*2) {
				  		val irow = (i + yoff) * inwidth;
				  		val irow2 = (i >> 1) * width;
				  		var j = 0;
				  		while (j < width*2) {
				  			val ii = irow + j + xoff;
				  			val jj = irow2 + (j >> 1);
				  			val r = math.max(colormap(buffer(ii) & 0xff), colormap(buffer2(ii) & 0xff));
				  			odata(jj) += r*cc;
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
		      case 4 => {                                // Pool a colormapped image
			if (colormap.asInstanceOf[AnyRef] == null) colormap = createColormap;	
		        while (i < height) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + j;
		      			odata(jj) = math.max(colormap(buffer(ii) & 0xff), colormap(buffer2(ii) & 0xff));
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
		      case 4 => {                                // Shrink a colormapped image
		      	out.clear;
		      	val cc = 1f/4;
			if (colormap.asInstanceOf[AnyRef] == null) colormap = createColormap;
		      	while (i < height*2) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = (i >> 1) * width;
		      		var j = 0;
		      		while (j < width*2) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + (j >> 1);
		      			val r = colormap(buffer(ii) & 0xff) * cc;
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
		      case 4 => {                                    // Copy a colormapped image
                         if (colormap.asInstanceOf[AnyRef] == null) colormap = createColormap;
		         while (i < height) {
		      		val irow = (i + yoff) * inwidth;
		      		val irow2 = i * width;
		      		var j = 0;
		      		while (j < width) {
		      			val ii = irow + j + xoff;
		      			val jj = irow2 + j;
		      			odata(jj) = colormap(buffer(ii) & 0xff);
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
  
  def stepx(action:Int, out0:FMat):(FMat, Float, Boolean, Int) = {
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
  	val nlives = lives();
  	(out, reward, done, nlives)
  };

  def stepx(action:Int):(FMat, Float, Boolean, Int) = stepx(action, null);

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
  		if (fire_to_start) act(1);
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
	
	final val ALEimageModeRaw = 0;
	final val ALEimageModeGrayscale = 1;
	final val ALEimageModeRGB = 2;
	final val ALEimageModeBinarize = 3;
	final val ALEimageModeColormap = 4;
	
	// 0 = Raw, 1 = Grayscale, 2 = RGB, 3 = binarize, 4 = colormap

	def stepAll(envs:Array[ALE],  actions:IMat, obs0:Array[FMat], rewards0:FMat, dones0:FMat):(Array[FMat], FMat, FMat) = {
		val npar = envs.length;
		val obs = if (obs0.asInstanceOf[AnyRef] == null) new Array[FMat](npar) else obs0;
		val rewards = if (rewards0.asInstanceOf[AnyRef] == null) zeros(1, npar) else rewards0;
		val dones = if (dones0.asInstanceOf[AnyRef] == null) zeros(1, npar) else dones0;
		(0 until npar).par.foreach((i) => {
			val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
			var reward = 0f;
			for (j <- 0 until nsteps) {
			  reward += envs(i).act(actions(i));
			  if (envs(i).pool & j == nsteps-2) envs(i).buffer2 = envs(i).getBufferData(envs(i).buffer2);
			}
			rewards(i) = reward;
			envs(i).buffer = envs(i).getBufferData(envs(i).buffer);
			obs(i) = envs(i).copyObs(obs(i));
			dones(i) = if (envs(i).game_over()) 1f else 0f;
			if (dones(i) == 1f) {
			  envs(i).reset_game();
			  if (envs(i).fire_to_start) {
			    rewards(i) += envs(i).act(1);
			  }
			}
		})
		(obs, rewards, dones)
	};

	def stepAll(envs:Array[ALE], actions:IMat):(Array[FMat], FMat, FMat) = stepAll(envs, actions, null, null, null);
	
  def stepAllx(envs:Array[ALE],  actions:IMat, obs0:Array[FMat], rewards0:FMat, dones0:FMat, lives0:FMat):(Array[FMat], FMat, FMat, FMat) = {
		val npar = envs.length;
		val obs = if (obs0.asInstanceOf[AnyRef] == null) new Array[FMat](npar) else obs0;
		val rewards = if (rewards0.asInstanceOf[AnyRef] == null) zeros(1, npar) else rewards0;
		val dones = if (dones0.asInstanceOf[AnyRef] == null) zeros(1, npar) else dones0;
		val nlives = if (lives0.asInstanceOf[AnyRef] == null) zeros(1, npar) else lives0;
		(0 until npar).par.foreach((i) => {
			val old_lives = envs(i).lives();
			val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
			var reward = 0f;
			for (j <- 0 until nsteps) {
			  reward += envs(i).act(actions(i));
			  if (envs(i).pool & j == nsteps-2) envs(i).buffer2 = envs(i).getBufferData(envs(i).buffer2);
			}
			rewards(i) = reward;
			envs(i).buffer = envs(i).getBufferData(envs(i).buffer);
			obs(i) = envs(i).copyObs(obs(i));
			dones(i) = if (envs(i).game_over()) 1f else 0f;
			if (dones(i) == 1f) {
			  envs(i).reset_game();
			}
			val new_lives = envs(i).lives();
			if (envs(i).fire_to_start && (dones(i) == 1f || new_lives < old_lives)) {
				rewards(i) += envs(i).act(1);
			}
			nlives(i) = new_lives;
		})
		(obs, rewards, dones, nlives)
	};
	
	def getLives(envs:Array[ALE], nlives0:FMat):FMat = {
		val npar = envs.length;
		val nlives = if (nlives0.asInstanceOf[AnyRef] == null) zeros(1, npar) else nlives0;	  
		for (i <- 0 until npar) {
		  nlives(i) = envs(i).lives()
		}
		nlives;
	}

	def stepAllx(envs:Array[ALE], actions:IMat):(Array[FMat], FMat, FMat, FMat) = stepAllx(envs, actions, null, null, null, null);

	def stepAll2(envs:Array[ALE], actions:IMat, obs0:Array[Array[Byte]], rewards0:FMat, dones0:FMat):(Array[Array[Byte]], FMat, FMat) = {
		val npar = envs.length;
		val obs = if (obs0.asInstanceOf[AnyRef] == null) new Array[Array[Byte]](npar) else obs0;
		val rewards = if (rewards0.asInstanceOf[AnyRef] == null) zeros(1, npar) else rewards0;
		val dones = if (dones0.asInstanceOf[AnyRef] == null) zeros(1, npar) else dones0;
		(0 until npar).par.foreach((i) => {
		  val old_lives = envs(i).lives();
			val nsteps = envs(i).frameskip._1 + rg.nextInt(envs(i).frameskip._2 - envs(i).frameskip._1 + 1);
			rewards(i) = (0 until nsteps).map((j) => envs(i).act(actions(i))).sum;
			obs(i) = envs(i).getBufferData(envs(i).buffer);
			dones(i) = if (envs(i).game_over()) 1f else 0f;
			if (dones(i) == 1f) {
				envs(i).reset_game();
			}
			val new_lives = envs(i).lives();
			if (envs(i).fire_to_start && (dones(i) == 1f || new_lives < old_lives)) {
				rewards(i) += envs(i).act(1);
			}
		})
		(obs, rewards, dones)
	};

		def stepAll2(envs:Array[ALE], actions:IMat):(Array[Array[Byte]], FMat, FMat) = stepAll2(envs, actions, null, null, null);
}


