package edu.berkeley.bid;
import java.io.*;

public class ALE implements Serializable {

    static {
	LibUtils.loadLibrary("bidmachale", false);
    }

    private long handle = 0;

    public ALE() {
	newALE();
    }

    protected void finalize() {
        if (handle != 0) {
            deleteALE();
            handle = 0;
        }
    }

    private native int newALE();

    private native int deleteALE();

    public native int getInt(String str);

    public native float getFloat(String str);

    public native boolean getBoolean(String str);

    public native String getString(String str);

    public native int setInt(String str, int v);

    public native int setFloat(String str, float v);

    public native int setBoolean(String str, boolean v);

    public native int setString(String str, String v);

    public native int loadROM(String str);

    public native int [] getLegalActionSet();

    public native int [] getMinimalActionSet();

    public native float act(int action);

    public native boolean game_over();

    public native int reset_game();

    public native int [] getScreenDims();

    public native int getScreenSize();

    public native byte [] getScreenData(byte [] data);

    public native byte [] getScreenRGB(byte [] data);

    public native byte [] getScreenGrayscale(byte [] data);

}
