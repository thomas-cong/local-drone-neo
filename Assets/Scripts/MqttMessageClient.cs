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

    [SerializeField] private Color connectedVoxelColor = Color.green;
    [SerializeField] private Transform anchorTransform;
    [SerializeField] private VoxelRenderSettings voxelSettings;

    public bool IsConnected => client != null && client.IsConnected;
    public bool ConnectionAttempted { get; private set; }
    public bool ConnectionSuccessful { get; private set; }

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
        if (string.IsNullOrEmpty(LatestPayload))
        {
            return;
        }

        if (!MqttPayloadDecoder.TryDecode(LatestPayload, out var payload) || payload?.DepthMatrix == null || payload.SegmentationMatrix == null)
        {
            return;
        }

        int rows = payload.DepthMatrix.Count;
        int cols = rows > 0 ? payload.DepthMatrix[0].Count : 0;
        if (rows == 0 || cols == 0)
        {
            return;
        }

        var depth = new float[rows, cols];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                depth[r, c] = payload.DepthMatrix[r][c];
            }
        }

        var maskInts = MatrixUtility.ToBinaryMask(ToFloatMatrix(payload.SegmentationMatrix));
        MatrixUtility.ApplyBinaryMaskInPlace(depth, maskInts);

        const int step = 3;
        VoxelRenderer.Clear();
        var anchor = AnchorPosition;
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
        float heightScale = condensed ? voxelSettings.condensedHeightScale : 0.005f;
        float voxelSize = condensed ? voxelSettings.condensedVoxelSize : 0.015f;
        int considered = 0;
        int rendered = 0;
        for (int r = 0; r < rows; r += step)
        {
            for (int c = 0; c < cols; c += step)
            {
                considered++;
                if (maskInts[r, c] == 0)
                {
                    continue;
                }

                float value = depth[r, c];
                if (value <= 0f)
                {
                    continue;
                }

                rendered++;
                float verticalOffset = (r - (rows - 1) * 0.5f) * spacing;
                float horizontalOffset = (c - (cols - 1) * 0.5f) * spacing;
                var position = anchor + baseOffset + right * horizontalOffset + upDir * verticalOffset + forward * (value * heightScale);
                VoxelRenderer.RenderVoxel(position, voxelSize, Color.red);
                if (rendered <= 5)
                {
                    Debug.Log($"[Voxel] Sample at row {r}, col {c}, depth {value:F2}, pos {position}");
                }
            }
        }

        float proportion = considered > 0 ? (float)rendered / considered : 0f;
        Debug.Log($"[Voxel] Rendered {rendered}/{considered} samples ({proportion:P1}) using spacing {spacing:F4}, heightScale {heightScale:F4}, voxelSize {voxelSize:F4} (condensed={condensed})");

        if (voxelSettings != null && voxelSettings.showTestGrid)
        {
            RenderTestGrid(anchor, condensed);
        }
    }

    private Vector3 AnchorPosition
    {
        get
        {
            if (anchorTransform != null)
            {
                return anchorTransform.position;
            }

            var cam = Camera.main;
            return cam != null ? cam.transform.position : Vector3.zero;
        }
    }

    private Vector3 AnchorForward
    {
        get
        {
            if (anchorTransform != null)
            {
                return anchorTransform.forward;
            }

            var cam = Camera.main;
            return cam != null ? cam.transform.forward : Vector3.forward;
        }
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
        float heightScale = condensed ? voxelSettings.condensedHeightScale : 0.005f;
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
                var position = anchor + offset + right * horizontalOffset + upDir * verticalOffset;
                VoxelRenderer.RenderVoxel(position, voxelSize, Color.green);
            }
        }
    }
}
