package org.tensorflow.lite.examples.classification.arcore;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.PixelCopy;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Config;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.SharedCamera;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.MaterialFactory;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;

import org.checkerframework.checker.units.qual.C;
import org.tensorflow.lite.examples.classification.CameraActivity;
import org.tensorflow.lite.examples.classification.ClassifierActivity;
import org.tensorflow.lite.examples.classification.R;
import org.w3c.dom.Text;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Objects;

import androidx.fragment.app.Fragment;

public class ARCoreActivity extends AppCompatActivity {
    private Session session;
    private SharedCamera sharedCamera;
    private String cameraId;
    private ModelRenderable modelRenderable;
    private ArFragment arFragment;
    private ArrayList<Anchor> placedAnchors = new ArrayList<>();
    private ArrayList<AnchorNode> placedAnchorNodes = new ArrayList<>();
    private TextView distanceText;
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
        Button clearAllButton = findViewById(R.id.clearButton);
        Button AIButton = findViewById(R.id.AIButton);

        clearAllButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                clearAllText();
                clearAllAnchors();
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
        final Bitmap bitmap = Bitmap.createBitmap(view.getWidth(), view.getHeight(), Bitmap.Config.ARGB_8888);
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