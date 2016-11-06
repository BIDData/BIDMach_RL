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

static ALEInterface * getALE(JNIEnv *env, jclass clazz, jobject jale)
{
  jfieldID handle_id = env->GetFieldID(clazz, "handle", "J");
  jlong handle = env->GetLongField(jale, handle_id);
  ALEInterface *alep = (ALEInterface *)long2void(handle);
  return alep;
}

static void setALE(JNIEnv *env, jclass clazz, jobject jale, ALEInterface *alep)
{
  jfieldID handle_id = env->GetFieldID(clazz, "handle", "J");
  jlong handle = void2long(alep);
  env->SetLongField(jale, handle_id, handle);
}


extern "C" {

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_RAND_createALE
(JNIEnv *env, jclass clazz, jobject jale)
{
  ALEInterface * alep = new ALEInterface();
  int status = (alep != NULL);
  setALE(env, clazz, jale, alep);
  
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_RAND_delALE
(JNIEnv *env, jclass clazz, jobject jale)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  delete [] alep;
  setALE(env, clazz, jale, NULL);
  
  return 0;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_getInt
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  jint retval = alep -> getInt(vname);
  env -> ReleaseStringUTFChars(jvname, vname);
  return retval;
}

JNIEXPORT jfloat JNICALL Java_edu_berkeley_bid_ALE_getFloat
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  jfloat retval = alep -> getFloat(vname);
  env -> ReleaseStringUTFChars(jvname, vname);
  return retval;
}

JNIEXPORT jboolean JNICALL Java_edu_berkeley_bid_ALE_getBoolean
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  jboolean retval = alep -> getBool(vname);
  env -> ReleaseStringUTFChars(jvname, vname);
  return retval;
}

JNIEXPORT jstring JNICALL Java_edu_berkeley_bid_ALE_getString
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  std::string str = alep -> getString(vname);
  jstring jstr = env->NewStringUTF(str.c_str());
  env -> ReleaseStringUTFChars(jvname, vname);
  return jstr;
}


JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setInt
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname, jint ival)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  alep -> setInt(vname, ival);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setFloat
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname, jfloat fval)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  alep -> setFloat(vname, fval);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setBoolean
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname, jboolean bval)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  alep -> setBool(vname, bval);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_setString
(JNIEnv *env, jclass clazz, jobject jale, jstring jvname, jstring jval)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  int status = (alep != NULL);
  char *vname = (char *)(env->GetStringUTFChars(jvname, 0));
  char *vval = (char *)(env->GetStringUTFChars(jval, 0));
  alep -> setString(vname, vval);
  env -> ReleaseStringUTFChars(jval, vval);
  env -> ReleaseStringUTFChars(jvname, vname);
  return status;
}

JNIEXPORT jint JNICALL Java_edu_berkeley_bid_ALE_loadROM
(JNIEnv *env, jclass clazz, jobject jale, jstring jromname)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  int status = (alep != NULL);
  char *romname = (char *)(env->GetStringUTFChars(jromname, 0));
  alep -> loadROM(romname);
  env -> ReleaseStringUTFChars(jromname, romname);
  return status;
}

JNIEXPORT jintArray JNICALL Java_edu_berkeley_bid_ALE_getLegalActionSet
(JNIEnv *env, jclass clazz, jobject jale)
{
  int i, size;
  ALEInterface *alep = getALE(env, clazz, jale);
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

JNIEXPORT jfloat JNICALL Java_edu_berkeley_bid_ALE_act
(JNIEnv *env, jclass clazz, jobject jale, jint action)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  float reward = alep->act((Action)action);
  return reward;
}

JNIEXPORT jboolean JNICALL Java_edu_berkeley_bid_ALE_game_1over
(JNIEnv *env, jclass clazz, jobject jale)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  jboolean done = alep->game_over();  
  return done;
}

JNIEXPORT jboolean JNICALL Java_edu_berkeley_bid_ALE_reset_1game
(JNIEnv *env, jclass clazz, jobject jale)
{
  ALEInterface *alep = getALE(env, clazz, jale);
  int status = (alep != NULL);
  alep->reset_game();
  return status;
}

}
