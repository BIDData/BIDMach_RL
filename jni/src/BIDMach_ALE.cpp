#include <jni.h>
#include <ale_interface.hpp>


union VoidLong {
  jlong l;
  void* p;
};

static jlong void2long(void* ptr) {
  union VoidLong v;
  v.l = (jlong) 0; 
  v.p = ptr;
  return v.l;
}

static void* long2void(jlong l) {
  union VoidLong v;
  v.l = l;
  return v.p;
}

static ALEInterface * getALE(JNIEnv *env, jobject jale)
{
  jclass clazz = env->GetObjectClass(jale);
  jfieldID handle_id = env->GetFieldID(clazz, "handle", "J");
  jlong handle = env->GetLongField(jale, handle_id);
  ALEInterface *alep = (ALEInterface *)long2void(handle);
  return alep;
}

static void setALE(JNIEnv *env, jobject jale, ALEInterface *alep)
{
  jclass clazz = env->GetObjectClass(jale);
  jfieldID handle_id = env->GetFieldID(clazz, "handle", "J");
  jlong handle = void2long(alep);
  env->SetLongField(jale, handle_id, handle);
}


extern "C" {

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_newALE
(JNIEnv *env, jobject jale)
{
  ALEInterface * alep = new ALEInterface(2);
  int status = (alep != NULL);
  setALE(env, jale, alep);
  
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_deleteALE
(JNIEnv *env, jobject jale)
{
  ALEInterface *alep = getALE(env, jale);
  delete [] alep;
  setALE(env, jale, NULL);
  
  return 0;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_getInt
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  jint retval = alep -> getInt(vname);
  env -> ReleaseStringUTFChars(jvname, vname);
  return retval;
}

JNIEXPORT jfloat JNICALL Java_edu_berkeley_bid_ALE_getFloat
(JNIEnv *env, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  jfloat retval = alep -> getFloat(vname);
  env -> ReleaseStringUTFChars(jvname, vname);
  return retval;
}

JNIEXPORT jboolean JNICALL Java_edu_berkeley_bid_ALE_getBoolean
(JNIEnv *env, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  jboolean retval = alep -> getBool(vname);
  env -> ReleaseStringUTFChars(jvname, vname);
  return retval;
}

JNIEXPORT jstring JNICALL Java_edu_berkeley_bid_ALE_getString
(JNIEnv *env, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  std::string str = alep -> getString(vname);
  jstring jstr = env->NewStringUTF(str.c_str());
  env -> ReleaseStringUTFChars(jvname, vname);
  return jstr;
}


JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setInt
(JNIEnv *env, jobject jale, jstring jvname, jint ival)
{
  ALEInterface *alep = getALE(env, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  alep -> setInt(vname, ival);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setFloat

(JNIEnv *env, jobject jale, jstring jvname, jfloat fval)
{
  ALEInterface *alep = getALE(env, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  alep -> setFloat(vname, fval);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setBoolean
(JNIEnv *env, jobject jale, jstring jvname, jboolean bval)
{
  ALEInterface *alep = getALE(env, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  alep -> setBool(vname, bval);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setString
(JNIEnv *env, jobject jale, jstring jvname, jstring jval)
{
  ALEInterface *alep = getALE(env, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  char *vval = (char *)(env->GetStringUTFChars(jval, 0));
  alep -> setString(vname, vval);
  env -> ReleaseStringUTFChars(jval, vval);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_loadROM
(JNIEnv *env, jobject jale, jstring jromname)
{
  ALEInterface *alep = getALE(env, jale);
  int status = (alep != NULL);
  char *romname = (char *)(env->GetStringUTFChars(jromname, 0));
  alep -> loadROM(romname);
  env -> ReleaseStringUTFChars(jromname, romname);
  return status;
}

JNIEXPORT jintArray JNICALL Java_edu_berkeley_bid_ALE_getLegalActionSet
(JNIEnv *env, jobject jale)
{
  int i, size;
  ALEInterface *alep = getALE(env, jale);
  std::vector<Action> legal_actions = alep->getLegalActionSet();
  size = legal_actions.size();
  jintArray result = env->NewIntArray(size);
  if (result == NULL) {
    return NULL; 
  }
  jint *body = env->GetIntArrayElements(result, 0);
  for (i = 0; i < size; i++) {
    body[i] = legal_actions[i];
  }
  env->ReleaseIntArrayElements(result, body, 0);
  return result;
}

JNIEXPORT jintArray JNICALL Java_edu_berkeley_bid_ALE_getMinimalActionSet
(JNIEnv *env, jobject jale)
{
  int i, size;
  ALEInterface *alep = getALE(env, jale);
  std::vector<Action> legal_actions = alep->getMinimalActionSet();
  size = legal_actions.size();
  jintArray result = env->NewIntArray(size);
  if (result == NULL) {
    return NULL; 
  }
  jint *body = env->GetIntArrayElements(result, 0);
  for (i = 0; i < size; i++) {
    body[i] = legal_actions[i];
  }
  env->ReleaseIntArrayElements(result, body, 0);
  return result;
}

JNIEXPORT jfloat JNICALL Java_edu_berkeley_bid_ALE_act
(JNIEnv *env, jobject jale, jint action)
{
  ALEInterface *alep = getALE(env, jale);
  float reward = alep->act((Action)action);
  return reward;
}

JNIEXPORT jboolean JNICALL Java_edu_berkeley_bid_ALE_game_1over
(JNIEnv *env, jobject jale)
{
  ALEInterface *alep = getALE(env, jale);
  jboolean done = alep->game_over();  
  return done;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_reset_1game
(JNIEnv *env, jobject jale)
{
  ALEInterface *alep = getALE(env, jale);
  int status = (alep != NULL);
  alep->reset_game();
  return status;
}

JNIEXPORT jintArray JNICALL Java_edu_berkeley_bid_ALE_getScreenDims
(JNIEnv *env, jobject jale)
{
  ALEInterface *alep = getALE(env, jale);
  jintArray result = env->NewIntArray(3);
  if (result == NULL) {
    return NULL; 
  }
  jint *body = env->GetIntArrayElements(result, 0);
  const ALEScreen& screen= alep->getScreen();
  body[0] = screen.width();
  body[1] = screen.height();
  body[2] = sizeof(pixel_t);
  env->ReleaseIntArrayElements(result, body, 0);
  return result;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_getScreenSize
(JNIEnv *env, jobject jale)
{
  ALEInterface *alep = getALE(env, jale);
  if (alep == NULL) {
    return -1;
  }
  const ALEScreen& screen= alep->getScreen();
  int size = screen.arraySize();
  return size;
}

JNIEXPORT jbyteArray JNICALL Java_edu_berkeley_bid_ALE_getScreenData
(JNIEnv *env, jobject jale, jbyteArray jdata)
{
  int i;
  ALEInterface *alep = getALE(env, jale);
  if (alep == NULL) {
    return NULL;
  }
  const ALEScreen& screen= alep->getScreen();
  int size = screen.arraySize();
  if (jdata == NULL) {
    jdata = env->NewByteArray(size);
  }
  if (jdata == NULL) {
    return NULL; 
  }
  jbyte *data = env->GetByteArrayElements(jdata, 0);
  if (data == NULL) {
    return NULL;
  }
  if (env->GetArrayLength(jdata) != size) {
    return NULL;
  }
  jbyte *screendata = (jbyte *)screen.getArray();
  for (i = 0; i < size; i++) {
    data[i] = screendata[i];
  }
  env->ReleaseByteArrayElements(jdata, data, 0);
  return jdata;
}

JNIEXPORT jbyteArray JNICALL Java_edu_berkeley_bid_ALE_getScreenRGB
(JNIEnv *env, jobject jale, jbyteArray jdata)
{
  int i;
  ALEInterface *alep = getALE(env, jale);
  if (alep == NULL) {
    return NULL;
  }
  const ALEScreen& screen= alep->getScreen();
  int size = screen.arraySize();
  if (jdata == NULL) {
    jdata = env->NewByteArray(size*3);
  }
  if (jdata == NULL) {
    return NULL; 
  }
  jbyte *data = env->GetByteArrayElements(jdata, 0);
  if (data == NULL) {
    return NULL;
  }
  if (env->GetArrayLength(jdata) != 3*size) {
    return NULL;
  }
  jbyte *screendata = (jbyte *)screen.getArray();
  alep->theOSystem->colourPalette().applyPaletteRGB((unsigned char *)data, (unsigned char *)screendata, size);
  env->ReleaseByteArrayElements(jdata, data, 0);
  return jdata;
}

JNIEXPORT jbyteArray JNICALL Java_edu_berkeley_bid_ALE_getScreenGrayscale
(JNIEnv *env, jobject jale, jbyteArray jdata)
{
  int i;
  ALEInterface *alep = getALE(env, jale);
  if (alep == NULL) {
    return NULL;
  }
  const ALEScreen& screen= alep->getScreen();
  int size = screen.arraySize();
  if (jdata == NULL) {
    jdata = env->NewByteArray(size);
  }
  if (jdata == NULL) {
    return NULL; 
  }
  jbyte *data = env->GetByteArrayElements(jdata, 0);
  if (data == NULL) {
    return NULL;
  }
  if (env->GetArrayLength(jdata) != size) {
    return NULL;
  }
  jbyte *screendata = (jbyte *)screen.getArray();
  alep->theOSystem->colourPalette().applyPaletteGrayscale((unsigned char *)data, (unsigned char *)screendata, size);
  env->ReleaseByteArrayElements(jdata, data, 0);
  return jdata;
}

JNIEXPORT jbyteArray JNICALL Java_edu_berkeley_bid_ALE_getScreenPaletteRGB
(JNIEnv *env, jobject jale, jbyteArray jdata)
{
  int i;
  ALEInterface *alep = getALE(env, jale);
  if (alep == NULL) {
    return NULL;
  }
  int dsize = 256;
  jbyte *screendata = new jbyte[dsize];
  for (i = 0; i < size; i++) {
    screendata[i] = i;
  }
  if (jdata == NULL) {
    jdata = env->NewByteArray(dsize*3);
  }
  if (jdata == NULL) {
    return NULL; 
  }
  jbyte *data = env->GetByteArrayElements(jdata, 0);
  if (data == NULL) {
    return NULL;
  }
  if (env->GetArrayLength(jdata) != 3*size) {
    return NULL;
  }
  alep->theOSystem->colourPalette().applyPaletteRGB((unsigned char *)data, (unsigned char *)screendata, size);
  delete [] screendata;
  env->ReleaseByteArrayElements(jdata, data, 0);
  return jdata;
}


}
