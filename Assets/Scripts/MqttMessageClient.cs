using System;
using System.Collections.Generic;
using System.Text;
using M2MqttUnity;
using uPLibrary.Networking.M2Mqtt.Messages;
using UnityEngine;

public class MqttMessageClient : M2MqttUnityClient
{
    [SerializeField]
    private string topic = "depth/seg";

    private readonly object _payloadLock = new object();
    private string _latestPayload;
    private bool _hasPendingPayload;

    [SerializeField] private Color connectedVoxelColor = Color.green;
    [SerializeField] private Transform anchorTransform;
    [SerializeField] private Transform cameraTransformOverride;
    [SerializeField] private OVRInput.Button anchorButton = OVRInput.Button.SecondaryIndexTrigger;
    [SerializeField] private float joystickMoveSpeed = 0.05f;
    [SerializeField] private VoxelRenderSettings voxelSettings;
    [SerializeField] private bool logFrameEveryTen = true;

    public bool IsConnected => client != null && client.IsConnected;
    public bool ConnectionAttempted { get; private set; }
    public bool ConnectionSuccessful { get; private set; }
    private bool anchorLocked;
    private Vector3 lockedAnchorPosition;
    private Vector3 lockedAnchorForward = Vector3.forward;
    private Vector3 manualOffset = Vector3.zero;

    public string LatestPayload
    {
        get
        {
            lock (_payloadLock)
            {
                return _latestPayload;
            }
        }
        private set
        {
            lock (_payloadLock)
            {
                _latestPayload = value;
            }
        }
    }

    protected override void Awake()
    {
        base.Awake();
        if (anchorTransform != null)
        {
            CaptureAnchor(anchorTransform.position, anchorTransform.forward);
        }
    }

    protected override void Start()
    {
        base.Start();
        if (anchorTransform != null)
        {
            CaptureAnchor(anchorTransform.position, anchorTransform.forward);
        }
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        if (anchorTransform != null)
        {
            CaptureAnchor(anchorTransform.position, anchorTransform.forward);
        }
    }
#endif

    protected override void OnConnecting()
    {
        base.OnConnecting();
        ConnectionAttempted = true;
        Debug.Log($"[MQTT] Attempting to connect to {brokerAddress}:{brokerPort}...");
        Debug.Log($"[MQTT] Timeout set to {timeoutOnConnection}ms");
    }

    protected override void OnConnected()
    {
        base.OnConnected();
        ConnectionSuccessful = true;
        Debug.Log($"[MQTT] Successfully connected to {brokerAddress}:{brokerPort}");
        RenderConnectedVoxel();
        RenderDepthMatrixVoxels();
    }

    protected override void Update()
    {
        base.Update();
        if (OVRInput.GetDown(anchorButton, OVRInput.Controller.All))
        {
            SetAnchorToCamera();
        }

        Vector2 primaryStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
        if (primaryStick.sqrMagnitude > 0.0001f)
        {
            manualOffset += (AnchorRight * primaryStick.x + AnchorUp * primaryStick.y) * joystickMoveSpeed * Time.deltaTime;
        }

        Vector2 secondaryStick = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick);
        if (secondaryStick.sqrMagnitude > 0.0001f)
        {
            manualOffset += AnchorForward * secondaryStick.y * joystickMoveSpeed * Time.deltaTime;
        }
    }

    protected override void OnConnectionFailed(string errorMessage)
    {
        base.OnConnectionFailed(errorMessage);
        ConnectionSuccessful = false;
        VoxelRenderer.Clear();
        Debug.LogError($"[MQTT] Connection failed: {errorMessage}");
        Debug.LogError($"[MQTT] Please check:");
        Debug.LogError($"  - Is broker running at {brokerAddress}:{brokerPort}?");
        Debug.LogError($"  - Is network accessible?");
        Debug.LogError($"  - Are credentials correct? (user: {(string.IsNullOrEmpty(mqttUserName) ? "none" : mqttUserName)})");
    }

    protected override void OnConnectionLost()
    {
        base.OnConnectionLost();
        ConnectionSuccessful = false;
        VoxelRenderer.Clear();
        Debug.LogWarning($"[MQTT] Connection lost to {brokerAddress}:{brokerPort}");
    }

    protected override void OnDisconnected()
    {
        base.OnDisconnected();
        VoxelRenderer.Clear();
    }

    protected override void SubscribeTopics() {
        if (client == null || string.IsNullOrEmpty(topic))
         {
             Debug.LogWarning("[MQTT] Topic is missing, cannot subscribe.");
             return;
         }

         client.Subscribe(new[] { topic }, new[] { MqttMsgBase.QOS_LEVEL_AT_LEAST_ONCE });
         Debug.Log($"[MQTT] Subscribed to topic: {topic}");
    }

    protected override void UnsubscribeTopics(){
        if (client == null || string.IsNullOrEmpty(topic))
        {
            return;
        }

        client.Unsubscribe(new[] { topic });
    }

    protected override void DecodeMessage(string topic, byte[] message) {
        base.DecodeMessage(topic, message);

        string payload = Encoding.UTF8.GetString(message);
        LatestPayload = payload;
        _hasPendingPayload = true;
        Debug.Log($"[MQTT] Message received on {topic} ({message.Length} bytes)");

        if (payload.Length <= 200)
        {
            Debug.Log($"[MQTT] Payload: {payload}");
        }
        else
        {
            Debug.Log($"[MQTT] Payload (truncated): {payload.Substring(0, 200)}...");
        }

        RenderDepthMatrixVoxels();
    }

    private void RenderConnectedVoxel()
    {
        VoxelRenderer.RenderVoxel(Vector3.zero, 0.05f, connectedVoxelColor);
    }

    private void RenderDepthMatrixVoxels()
    {
        if (!_hasPendingPayload || string.IsNullOrEmpty(LatestPayload))
        {
            return;
        }

        _hasPendingPayload = false;

        if (!MqttPayloadDecoder.TryDecode(LatestPayload, out var payload) || payload?.DepthMatrix == null || payload.SegmentationMatrix == null)
        {
            return;
        }

        if (logFrameEveryTen && payload.Frame % 10 == 0)
        {
            Debug.Log($"[Voxel] Rendering payload frame {payload.Frame}");
        }

        int rows = payload.DepthMatrix.Count;
        int cols = rows > 0 ? payload.DepthMatrix[0].Count : 0;
        if (rows == 0 || cols == 0)
        {
            return;
        }

        var depth = ToFloatMatrix(payload.DepthMatrix);

        var segFloat = ToFloatMatrix(payload.SegmentationMatrix);
        if (voxelSettings != null && voxelSettings.blurMaskBeforeRender)
        {
            MatrixUtility.BoxBlurInPlace(segFloat, 1);
        }
        var maskInts = MatrixUtility.ToBinaryMask(segFloat);
        MatrixUtility.ApplyBinaryMaskInPlace(depth, maskInts);

        int upscale = Mathf.Max(1, voxelSettings != null ? voxelSettings.renderMatrixUpscale : 1);
        int step = Mathf.Max(1, 3 / upscale);
        var depthUpscaled = upscale > 1 ? MatrixUtility.UpscaleMatrix(depth, upscale) : depth;
        var maskUpscaled = upscale > 1 ? MatrixUtility.UpscaleMask(maskInts, upscale) : maskInts;
        var anchor = AnchorPosition + manualOffset;
        var forward = AnchorForward.normalized;
        var right = Vector3.Normalize(Vector3.Cross(Vector3.up, forward));
        if (right == Vector3.zero)
        {
            right = Vector3.right;
        }
        var upDir = Vector3.Cross(forward, right);
        var baseOffset = forward * 0.5f;
        bool condensed = voxelSettings != null && voxelSettings.condensedView;
        if (voxelSettings == null)
        {
            Debug.LogWarning("[Voxel] No VoxelRenderSettings assigned; using default spacing.");
        }
        float baseSpacing = condensed ? voxelSettings.condensedSpacing : 0.02f;
        float spacing = baseSpacing * step;
        float heightScale = condensed ? voxelSettings.condensedHeightScale : voxelSettings != null ? voxelSettings.depthScale : 0.005f;
        float voxelSize = condensed ? voxelSettings.condensedVoxelSize : 0.015f;
        int considered = 0;
        int rendered = 0;
        bool useTemporal = voxelSettings != null && voxelSettings.enableTemporalBlend;
        float blend = voxelSettings != null ? Mathf.Clamp01(voxelSettings.blendFactor) : 0f;
        bool useBillboards = voxelSettings != null && voxelSettings.renderingMode == RenderingMode.Billboards;
        var voxels = new List<VoxelData>(rendered);
        var depthSamples = new List<float>();
        float minDepth = float.PositiveInfinity;
        float maxDepth = float.NegativeInfinity;
        for (int r = 0; r < rows; r += step)
        {
            for (int c = 0; c < cols; c += step)
            {
                if (maskInts[r, c] == 0)
                {
                    continue;
                }

                float value = depth[r, c];
                if (value <= 0f)
                {
                    continue;
                }

                minDepth = Mathf.Min(minDepth, value);
                maxDepth = Mathf.Max(maxDepth, value);
            }
        }
        for (int r = 0; r < depthUpscaled.GetLength(0); r += step)
        {
            for (int c = 0; c < depthUpscaled.GetLength(1); c += step)
            {
                considered++;
                if (maskUpscaled[r, c] == 0)
                {
                    continue;
                }

                float value = depthUpscaled[r, c];
                if (value <= 0f)
                {
                    continue;
                }

                rendered++;
                float verticalOffset = (r - (rows - 1) * 0.5f) * spacing;
                float horizontalOffset = (c - (cols - 1) * 0.5f) * spacing;
                var position = anchor + baseOffset + right * horizontalOffset - upDir * verticalOffset + forward * (value * heightScale);
                float depthT = Mathf.InverseLerp(minDepth, maxDepth, value);
                var color = Color.Lerp(Color.red * 0.4f, Color.red, 1f - depthT);
                voxels.Add(new VoxelData
                {
                    Position = position,
                    Size = voxelSize,
                    Color = color,
                    UseBillboard = useBillboards,
                    BillboardScale = voxelSettings != null ? voxelSettings.billboardScale : 0.02f,
                    BillboardTexture = voxelSettings != null ? voxelSettings.billboardTexture : null
                });
                depthSamples.Add(value);
            }
        }

        VoxelRenderer.ApplyFrame(voxels, useTemporal, blend);
        if (depthSamples.Count > 0)
        {
            depthSamples.Sort();
            float median = depthSamples[depthSamples.Count / 2];
            var textPosition = anchor + baseOffset + forward * (median * heightScale + 0.1f);
            VoxelRenderer.RenderText(textPosition, $"Median: {median:F2}", Color.red);
        }

        float proportion = considered > 0 ? (float)rendered / considered : 0f;
        if (voxelSettings != null && voxelSettings.showTestGrid)
        {
            RenderTestGrid(anchor, condensed);
        }
    }

    private Vector3 AnchorPosition
    {
        get
        {
            if (anchorLocked)
            {
                return lockedAnchorPosition;
            }

            if (anchorTransform != null)
            {
                return anchorTransform.position;
            }

            var head = GetHeadTransform();
            return head != null ? head.position : Vector3.zero;
        }
    }

    private Vector3 AnchorForward
    {
        get
        {
            if (anchorLocked)
            {
                return lockedAnchorForward;
            }

            if (anchorTransform != null)
            {
                return anchorTransform.forward;
            }

            var head = GetHeadTransform();
            return head != null ? head.forward : Vector3.forward;
        }
    }

    private Vector3 AnchorRight
    {
        get
        {
            var right = Vector3.Normalize(Vector3.Cross(Vector3.up, AnchorForward));
            return right == Vector3.zero ? Vector3.right : right;
        }
    }

    private Vector3 AnchorUp => Vector3.Normalize(Vector3.Cross(AnchorForward, AnchorRight));

    private void SetAnchorToCamera()
    {
        var head = GetHeadTransform();
        if (head == null)
        {
            Debug.LogWarning("[Voxel] No camera found to set anchor.");
            return;
        }

        CaptureAnchor(head.position, head.forward);

        if (anchorTransform == null)
        {
            var go = new GameObject("VoxelAnchor");
            anchorTransform = go.transform;
        }

        anchorTransform.SetParent(null);
        anchorTransform.position = lockedAnchorPosition;
        anchorTransform.rotation = Quaternion.LookRotation(lockedAnchorForward, Vector3.up);
        Debug.Log($"[Voxel] Anchor locked at {lockedAnchorPosition}, forward {lockedAnchorForward}");
    }

    private Transform GetHeadTransform()
    {
        if (cameraTransformOverride != null)
        {
            return cameraTransformOverride;
        }

        var cam = Camera.main;
        return cam != null ? cam.transform : null;
    }

    private void CaptureAnchor(Vector3 position, Vector3 forward)
    {
        lockedAnchorPosition = position;
        lockedAnchorForward = forward.sqrMagnitude > 0.0001f ? forward.normalized : Vector3.forward;
        anchorLocked = true;
        manualOffset = Vector3.zero;
    }

    private static float[,] ToFloatMatrix(List<List<int>> source)
    {
        int rows = source.Count;
        int cols = rows > 0 ? source[0].Count : 0;
        var result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result[r, c] = source[r][c];
            }
        }
        return result;
    }

    private void RenderTestGrid(Vector3 anchor, bool condensed)
    {
        const int size = 3;
        float spacing = condensed ? voxelSettings.condensedSpacing : 0.02f;
        float heightScale = condensed ? voxelSettings.condensedHeightScale : voxelSettings != null ? voxelSettings.depthScale : 0.005f;
        float voxelSize = condensed ? voxelSettings.condensedVoxelSize : 0.02f;
        var forward = AnchorForward.normalized;
        var right = Vector3.Normalize(Vector3.Cross(Vector3.up, forward));
        if (right == Vector3.zero)
        {
            right = Vector3.right;
        }
        var upDir = Vector3.Cross(forward, right);
        var offset = forward * 0.3f;

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                float verticalOffset = (i - (size - 1) * 0.5f) * spacing;
                float horizontalOffset = (j - (size - 1) * 0.5f) * spacing;
                var position = anchor + offset + right * horizontalOffset - upDir * verticalOffset;
                VoxelRenderer.RenderVoxel(position, voxelSize, Color.green);
            }
        }
    }
}
