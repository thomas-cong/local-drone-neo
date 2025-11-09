using System.Text;
using UnityEngine;
using TMPro;

public class MqttTextPresenter : MonoBehaviour
{
    [SerializeField] private MqttMessageClient mqttClient;
    [SerializeField] private TextMeshProUGUI targetText;
    [SerializeField] private bool showRawPayload = true;
    [SerializeField] private bool showDecodedSummary = true;
    [SerializeField] private int maxPayloadCharacters = 500;

    private readonly StringBuilder _builder = new StringBuilder(1024);
    private string _lastPayload;
    private bool _lastConnectionStatus;
    private bool _initialized;

    private void Reset()
    {
        targetText = GetComponent<TextMeshProUGUI>();
    }

    private void Update()
    {
        if (mqttClient == null || targetText == null)
        {
            return;
        }

        // Thread-safe read from the MQTT client
        var payload = mqttClient.LatestPayload;
        var isConnected = mqttClient.IsConnected;

        // Only update UI if something has changed
        // Always update on first frame to show initial state
        if (_initialized && payload == _lastPayload && isConnected == _lastConnectionStatus)
        {
            return;
        }

        _initialized = true;
        _lastPayload = payload;
        _lastConnectionStatus = isConnected;
        _builder.Clear();

        // Show connection status
        if (!mqttClient.ConnectionAttempted)
        {
            _builder.AppendLine("MQTT: Not connected");
        }
        else if (!mqttClient.ConnectionSuccessful)
        {
            _builder.AppendLine("MQTT: Connection failed!");
            _builder.AppendLine("Check console for details.");
        }
        else if (mqttClient.IsConnected)
        {
            _builder.AppendLine("MQTT: Connected");
        }
        else
        {
            _builder.AppendLine("MQTT: Disconnected");
        }

        _builder.AppendLine();

        if (string.IsNullOrEmpty(payload))
        {
            _builder.AppendLine("Waiting for MQTT payload...");
        }

        if (showDecodedSummary && MqttPayloadDecoder.TryDecode(payload, out var decoded))
        {
            _builder.AppendLine($"Frame: {decoded.Frame}");
            _builder.AppendLine($"Timestamp: {decoded.Timestamp:F3}");
            if (decoded.MatrixResolution != null)
            {
                _builder.AppendLine($"Resolution: {decoded.MatrixResolution.Width} x {decoded.MatrixResolution.Height}");
            }

            int depthRows = decoded.DepthMatrix?.Count ?? 0;
            int segRows = decoded.SegmentationMatrix?.Count ?? 0;
            _builder.AppendLine($"Depth rows: {depthRows}");
            _builder.AppendLine($"Segmentation rows: {segRows}");
        }

        if (showRawPayload && !string.IsNullOrEmpty(payload))
        {
            _builder.AppendLine("Payload:");
            _builder.AppendLine(TruncatePayload(payload));
        }

        targetText.text = _builder.ToString();
    }

    private string TruncatePayload(string payload)
    {
        if (string.IsNullOrEmpty(payload) || payload.Length <= maxPayloadCharacters)
        {
            return payload;
        }

        return payload.Substring(0, maxPayloadCharacters) + "...";
    }
}
