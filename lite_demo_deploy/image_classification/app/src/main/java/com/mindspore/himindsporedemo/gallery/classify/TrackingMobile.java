/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mindspore.himindsporedemo.gallery.classify;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * Call the MindSpore interface API in the Java layer.
 */
public class TrackingMobile {
    private final static String TAG = "TrackingMobile";

    static {
        try {
            System.loadLibrary("mlkit-label-MS");
            Log.i(TAG, "load libiMindSpore.so successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "UnsatisfiedLinkError " + e.getMessage());
        }
    }

    // The address of the running inference environment.
    private long netEnv = 0;

    private final Context mActivity;

    public TrackingMobile(Context activity) {
        this.mActivity = activity;
    }

    /**
     * JNI load model and also create model inference environment.
     *
     * @param modelBuffer Model buffer.
     * @param numThread The num of thread.
     * @return MindSpore Inference environment address.
     */
    public native long loadModel(ByteBuffer modelBuffer, int numThread);

    /**
     * Running model.
     * 
     * @param netEnv Inference environment address.
     * @param img A picture to be inferred.
     * @return Inference result
     */
    public native String runNet(long netEnv, Bitmap img);

    /**
     * Unbind model data.
     * 
     * @param netEnv Inference environment address.
     * @return Unbound state.
     */
    public native boolean unloadModel(long netEnv);

    /**
     * The C++ side is encapsulated into a method of the MSNetWorks class
     * 
     * @param modelPath Model file location
     * @return Load model file status
     */
    public boolean loadModelFromBuf(String modelPath) {
        ByteBuffer buffer = loadModelFile(modelPath);
        netEnv = loadModel(buffer, 2);  //numThread's default setting is 2.
        if (netEnv == 0){ // Loading model failed.
            return false;
        }

        return true;
    }

    /**
     * Run MindSpore inference.
     */
    public String MindSpore_runnet(Bitmap img) {
        String ret_str = runNet(netEnv, img);
        return ret_str;
    }

    /**
     * Unload model.
     * @return true
     */
    public boolean unloadModel() {
        unloadModel(netEnv);
        return true;
    }

    /**
     * Load model file stream.
     * @param modelPath Model file path.
     * @return  Model ByteBuffer.
     */
    public ByteBuffer loadModelFile(String modelPath) {
        InputStream is = null;
        try {
            is = mActivity.getAssets().open(modelPath);
            byte[] bytes = new byte[is.available()];
            is.read(bytes);
            return ByteBuffer.allocateDirect(bytes.length).put(bytes);
        } catch (Exception e) {
            Log.d("loadModelFile", " Exception occur. ");
            Log.e(TAG, Log.getStackTraceString(e));
        }
        return null;
    }
}
