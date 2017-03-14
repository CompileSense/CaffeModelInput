package com.compilesense.liuyi.caffemodelinput;

import android.Manifest;
import android.content.pm.PackageManager;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import com.compilesense.liuyi.caffemodelinput.caffecnn.ConvolutionLayer;
import com.compilesense.liuyi.caffemodelinput.messagepack.ModelInput;

public class MainActivity extends AppCompatActivity {
    public static String TAG = "MainActivity";
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    ModelInput modelInput;

    ConvolutionLayer convolutionLayer;

    private ICheckPermission iCheckPermission;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        findViewById(R.id.bt_release_layer).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (modelInput == null){
                    return;
                }
                modelInput.releaseLayers();
            }
        });

        findViewById(R.id.bt_test_ConvLayer_compute).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                ConvolutionLayer.testCompult();

//                NativeTest.testComputeIm2Col();
//                NativeTest.testComputeRelu();
//
//                NativeTest.testComputePrelu();
//                NativeTest.testComputeTanh();
//                stringFromJNI();

                NativeTest.testMathExp();
            }
        });

        // Example of a call to a native method
        Button in = (Button) findViewById(R.id.bt_input_model);
        in.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (modelInput != null){
                    return;
                }
                try {
                    modelInput = new ModelInput();
                    modelInput.readNetFileFromAssert(MainActivity.this,"Cifar10_def.txt");
                }catch (Exception e){
                    e.printStackTrace();
                }

//                if (iCheckPermission != null){
//                    return;
//                }else {
//                    iCheckPermission = new ICheckPermission() {
//                        @Override
//                        public void pass() {
//
//                            try {
//                                new ModelInput().readNetFileFromAssert(MainActivity.this,"CaffeNet_def.txt");
//                            }catch (Exception e){
//                                e.printStackTrace();
//                            }
//                        }
//
//                        @Override
//                        public void filed() {
//                            requestPermission();
//                        }
//                    };
//                }
//
//                checkPermissionAndDoNext();

            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == ICheckPermission.REQUEST_EXTERNAL_STORAGE){
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED){
                if (iCheckPermission!=null){
                    iCheckPermission.pass();
                }
                iCheckPermission = null;
            }else if (grantResults[0] == PackageManager.PERMISSION_DENIED){
                Log.e(TAG,"权限申请被拒绝");
            }
        }
    }

    private void requestPermission(){
        ActivityCompat.requestPermissions(this,
                new String[]{
                        Manifest.permission.WRITE_EXTERNAL_STORAGE},
                ICheckPermission.REQUEST_EXTERNAL_STORAGE);

        ActivityCompat.shouldShowRequestPermissionRationale(this,
                Manifest.permission.READ_CONTACTS);
    }

    private void checkPermissionAndDoNext(){
        Log.d(TAG,"checkPermissionAndDoNext");
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ){
            Log.d(TAG,"WRITE_EXTERNAL_STORAGE");
            iCheckPermission.filed();
        }
        iCheckPermission.pass();
        iCheckPermission = null;
    }

    private interface ICheckPermission{
        int REQUEST_EXTERNAL_STORAGE = 12;
        void pass();
        void filed();
    }



    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
