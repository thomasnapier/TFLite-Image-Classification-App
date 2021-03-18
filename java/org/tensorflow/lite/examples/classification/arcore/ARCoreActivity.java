package org.tensorflow.lite.examples.classification.arcore;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.PixelCopy;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.Anchor;
import com.google.ar.core.HitResult;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.SharedCamera;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.MaterialFactory;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.examples.classification.ClassifierActivity;
import org.tensorflow.lite.examples.classification.R;
import org.tensorflow.lite.examples.classification.tflite.Classifier;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.File;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public class ARCoreActivity extends AppCompatActivity {
    private static final int[] OUTPUT_WIDTH_TINY = new int[]{2535, 2535};
    private static final int[] OUTPUT_WIDTH_FULL = new int[]{10647, 10647};
    private static final int LABEL_SIZE = 1;
    private static final int INPUT_SIZE = 416;
    private Session session;
    private SharedCamera sharedCamera;
    private String cameraId;
    private ModelRenderable modelRenderable;
    private ArFragment arFragment;
    private ArrayList<Anchor> placedAnchors = new ArrayList<>();
    private ArrayList<AnchorNode> placedAnchorNodes = new ArrayList<>();
    private TextView distanceText;
    private TextView categoryText;
    private static boolean isRecursionEnabled = true;
    final int APP_PERMISSION_REQUEST = 7;

    private boolean installRequested = true;

    public ARCoreActivity() {
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_arcore_measurement);

        arFragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.sceneform_fragment);

        distanceText = findViewById(R.id.distanceText);
        categoryText = findViewById(R.id.categoryText);
        Button clearAllButton = findViewById(R.id.clearButton);
        Button AIButton = findViewById(R.id.AIButton);
        clearAllButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                clearAllText();
                clearAllAnchors();
                Bitmap image = captureImage();
                detectAbalone(image);
                //TensorImage tensorImage = convertToTensorImage(image);
                //detectObject(tensorImage);
                //classifyImage(tensorImage);
            }
        });

        arFragment.setOnTapArPlaneListener((hitResult, plane, motionEvent) -> {
            tapDistanceOf2Points(hitResult);
        });

        AIButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                changeActivity();
            }
        });
    }

    private void detectAbalone(Bitmap image){
        Interpreter.Options tfliteOptions = (new Interpreter.Options());
        Interpreter tfliteInterpreter = null;
        MappedByteBuffer tfliteModel = null;
        try {
            tfliteModel = FileUtil.loadMappedFile(this, "abalone.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(null != tfliteModel) {
            tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);
        }
        assert tfliteInterpreter != null;
        DataType imageDataType = tfliteInterpreter.getInputTensor(0).dataType();
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(416, 416, ResizeOp.ResizeMethod.BILINEAR))
//                .add(new QuantizeOp((float) 128.0, (float) 1/128))
                .build();
        TensorImage tensorImage = new TensorImage(imageDataType);
        tensorImage.load(image);
        tensorImage = imageProcessor.process(tensorImage);
        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 40560}, DataType.UINT8);
        tfliteInterpreter.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
        float[] values = probabilityBuffer.getFloatArray();
//        float result = probabilityBuffer.getFloatArray()[0];
//        result = result/255;
//        if(result < 0.5) {
//            categoryText.setText("Dog " + result);
//        }
//        else{
//            categoryText.setText("Cat " + result);
//        }

        for(Float value : values){
            Log.e("ACTUALVALUES", "This is the value: " + value);
        }
    }

    private void detectObject(TensorImage tensorImage){
        String modelFile = "abalone.tflite";
        ObjectDetector.ObjectDetectorOptions options = ObjectDetector.ObjectDetectorOptions.builder().setMaxResults(1).build();
        ObjectDetector objectDetector = null;
        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(this, modelFile, options);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model", e);
        }

        if (objectDetector != null) {
            List<Detection> results = objectDetector.detect(tensorImage);
            Log.e("tfliteSupport", "RESULTS: " + results);
        }
    }

    private void classifyImage(TensorImage tensorImage) {
//        Interpreter.Options tfliteOptions = (new Interpreter.Options());
//        @NonNull MappedByteBuffer tfliteModel = null;
//        try {
//            tfliteModel = FileUtil.loadMappedFile(this, "model.tflite");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        Interpreter tfliteInterpreter = new Interpreter(tfliteModel, tfliteOptions);
//        DataType dataType = tfliteInterpreter.getInputTensor(0).dataType();
//        TensorImage tensorImage = new TensorImage(dataType);
//        Bitmap image = captureImage();
//        tempPic.setImageBitmap(image);
//        tensorImage.load(image);
//        int[] output = tensorImage.getTensorBuffer().getShape();
//        tfliteInterpreter.run(tensorImage.getBuffer(), output);
//        tfliteInterpreter.close();
//        tfliteInterpreter = null;
//        System.out.println(output);

        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);

        Interpreter.Options tfliteOptions = (new Interpreter.Options());
        @NonNull MappedByteBuffer tfliteModel;
        Interpreter tflite = null;
        try {
            tfliteModel = FileUtil.loadMappedFile(this,  "model.tflite");
            tflite = new Interpreter(tfliteModel, tfliteOptions);
            System.out.println("This is the count of outputs: " + tflite.getOutputTensorCount());
            System.out.println("This is the count of inputs: " + tflite.getInputTensorCount());
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading model", e);
        }

        if(null != tflite){
            tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
        }

        float[] values = probabilityBuffer.getFloatArray();
//        float result = probabilityBuffer.getFloatArray()[0];
//        result = result/255;
//        if(result < 0.5) {
//            categoryText.setText("Dog " + result);
//        }
//        else{
//            categoryText.setText("Cat " + result);
//        }

        for(Float value : values){
            Log.e("ACTUALVALUES", "This is the value: " + value/255);
        }

        final String ASSOCIATED_AXIS_LABELS = "s.txt";
        List<String> associatedAxisLabels = null;
        try{
            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
        }catch (IOException e){
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        TensorProcessor probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

        if(null != associatedAxisLabels){
            TensorLabel labels = new TensorLabel(associatedAxisLabels, probabilityProcessor.process(probabilityBuffer));
            Map<String, Float> floatMap = labels.getMapWithFloatValue();
            for (Map.Entry<String, Float> entry : floatMap.entrySet()){
                String key = entry.getKey();
                Float value = entry.getValue();
                Log.e("tfliteResults", "Result is: " + key + ": " + value);
            }
        }
    }

    private TensorImage convertToTensorImage(Bitmap bitmap){
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(416, 416, ResizeOp.ResizeMethod.BILINEAR))
//                .add(new QuantizeOp((float) 128.0, (float) 1/128))
                .build();
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);
        tensorImage = imageProcessor.process(tensorImage);
        return tensorImage;
    }

    private void changeActivity(){
        Intent i = new Intent(this, ClassifierActivity.class);
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK|Intent.FLAG_ACTIVITY_CLEAR_TASK);
        this.startActivity(i);
    }

    private void setupPlane(HitResult hitResult){
        Anchor anchor = hitResult.createAnchor();
        placedAnchors.add(anchor);
        // Attach AnchorNode to an Anchor from the hit test
        AnchorNode anchorNode = new AnchorNode(anchor);
        placedAnchorNodes.add(anchorNode);
        anchorNode.setParent(arFragment.getArSceneView().getScene());
        createRenderable(anchorNode);
    }

    private void createRenderable(AnchorNode anchorNode){
        // Create 3D sphere using MaterialFactory and ModelRenderable to use as anchor markers
        MaterialFactory
                .makeOpaqueWithColor(this, new Color(android.graphics.Color.RED))
                .thenAccept(material -> {
                    ModelRenderable redSphereRenderable =
                            ShapeFactory.makeSphere(0.07f, new Vector3(0.0f, 0.15f, 0.0f), material);

                    createModel(anchorNode, redSphereRenderable);
                });
    }

    private void createModel(AnchorNode anchorNode, ModelRenderable modelRenderable){
        TransformableNode node = new TransformableNode(arFragment.getTransformationSystem());
        node.setParent(anchorNode);
        node.setRenderable(modelRenderable);
        node.select();
    }

    private void clearAllAnchors(){
        for(AnchorNode anchorNode : placedAnchorNodes){
            arFragment.getArSceneView().getScene().removeChild(anchorNode);
            Objects.requireNonNull(anchorNode.getAnchor()).detach();
            anchorNode.setParent(null);
            anchorNode.removeChild(anchorNode);
            anchorNode.setRenderable(null);
        }
        placedAnchors.clear();
        placedAnchorNodes.clear();
    }

    private void tapDistanceOf2Points(HitResult hitResult){
        // Allow up to 2 anchors and calculate distance on placement of the second marker
        if(placedAnchorNodes.size() == 0){
            setupPlane(hitResult);
        }
        else if(placedAnchorNodes.size() == 1){
            setupPlane(hitResult);
            measureDistanceOf2Points();
        }
        else{
            clearAllAnchors();
            clearAllText();
            Toast.makeText(this, "Cleared", Toast.LENGTH_SHORT).show();
        }
    }

    private void clearAllText() {
        distanceText.setText("");
    }

    private void updateDistanceText(Double distance) {
        distance = distance * 100; //convert to cm
        distanceText.setText(String.format("%.2fcm", distance));
    }

    private void measureDistanceOf2Points(){
        double distance = 0;
        distance = measureDistance(Objects.requireNonNull(placedAnchorNodes.get(0).getAnchor()).getPose(),
                Objects.requireNonNull(placedAnchorNodes.get(1).getAnchor()).getPose());
        updateDistanceText(distance);
        Log.i("Distance", "Distance is: " + distance);
    }

    private Double measureDistance(Pose objectPose1, Pose objectPose2){
        // Compute the difference vector between the two hit locations
        return calculateDistance(
                objectPose1.tx() - objectPose2.tx(),
                objectPose1.ty() - objectPose2.ty(),
                objectPose1.tz() - objectPose2.tz()
        );
    }

    private Double calculateDistance(Float x, Float y, Float z){
        // Compute the straight line distance between the two points
        return Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2) + Math.pow(z, 2));
    }

    private Bitmap captureImage(){
        ArSceneView view = arFragment.getArSceneView();
        final Bitmap bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
        PixelCopy.request(view, bitmap, (copyResult) -> {
            if(copyResult == PixelCopy.SUCCESS){
                //Do Nothing
            }
            else{
                Log.i("Image", "Capture Error");
            }
        }, new Handler(Looper.getMainLooper()));
        return bitmap;
    }
}