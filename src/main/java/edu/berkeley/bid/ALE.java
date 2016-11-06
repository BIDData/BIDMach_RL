package edu.berkeley.bid;

public final class ALE {

    static {
	LibUtils.loadLibrary("bidmachale", false);
    }

    private long handle = 0;

    public ALE() {
	newALE(this);
    }

    protected void finalize() {
        if (handle != 0) {
            deleteALE(this);
            handle = 0;
        }
    }

    private static native int newALE(ALE me);

    private static native int deleteALE(ALE me);

    private static native int getInt(ALE me, String str);

    private static native float getFloat(ALE me, String str);

    private static native boolean getBoolean(ALE me, String str);

    private static native String getString(ALE me, String str);

    private static native int setInt(ALE me, String str, int v);

    private static native int setFloat(ALE me, String str, float v);

    private static native int setBoolean(ALE me, String str, boolean v);

    private static native int setString(ALE me, String str, String v);

    private static native int loadROM(ALE me, String str);

    private static native int [] getLegalActionSet(ALE me);

    private static native int [] getMinimalActionSet(ALE me);

    private static native float act(ALE me, int action);

    private static native boolean game_over(ALE me);

    private static native int reset_game(ALE me);

    private static native int [] getScreenDims(ALE me);

    private static native int getScreenSize(ALE me);

    private static native byte [] getScreenData(ALE me, byte [] data);


    public int getInt(String str) {return getInt(this, str);}

    public float getFloat(String str) {return getFloat(this, str);}

    public boolean getBoolean(String str) {return getBoolean(this, str);}

    public String getString(String str) {return getString(this, str);}

    public int setInt(String str, int v) {return setInt(this, str, v);}

    public int setFloat(String str, float v) {return setFloat(this, str, v);}

    public int setBoolean(String str, boolean v) {return setBoolean(this, str, v);}

    public int setString(String str, String v) {return setString(this, str, v);}

    public int loadROM(String str) {return loadROM(this, str);}

    public int [] getLegalActionSet() {return getLegalActionSet(this);}

    public int [] getMinimalActionSet() {return getMinimalActionSet(this);}

    public float act(int action) {return act(this, action);}

    public boolean game_over()  {return game_over(this);}

    public int reset_game() {return reset_game(this);}

    public int [] getScreenDims() {return getScreenDims(this);}

    public int getScreenSize() {return getScreenSize(this);}

    public byte [] getScreenData(byte [] data) {return getScreenData(this, data);}

    
}
