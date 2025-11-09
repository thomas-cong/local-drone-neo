using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using UnityEngine;

public static class MqttPayloadDecoder
{
    public static bool TryDecode(string payload, out MqttPayload result)
    {
        result = null;

        if (string.IsNullOrEmpty(payload))
        {
            return false;
        }

        try
        {
            result = JsonConvert.DeserializeObject<MqttPayload>(payload);
            return result != null;
        }
        catch (JsonException ex)
        {
            Debug.LogWarning($"Failed to decode MQTT payload: {ex.Message}");
            return false;
        }
    }
}

[Serializable]
public sealed class MqttPayload
{
    [JsonProperty("frame")]
    public int Frame { get; set; }

    [JsonProperty("timestamp")]
    public float Timestamp { get; set; }

    [JsonProperty("matrix_resolution")]
    public MatrixResolution MatrixResolution { get; set; }

    [JsonProperty("depth_matrix")]
    public List<List<int>> DepthMatrix { get; set; }

    [JsonProperty("segmentation_matrix")]
    public List<List<int>> SegmentationMatrix { get; set; }
}

[Serializable]
public sealed class MatrixResolution
{
    [JsonProperty("width")]
    public int Width { get; set; }

    [JsonProperty("height")]
    public int Height { get; set; }
}
