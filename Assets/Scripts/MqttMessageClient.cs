using System;
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
    }

    protected override void OnConnectionFailed(string errorMessage)
    {
        base.OnConnectionFailed(errorMessage);
        ConnectionSuccessful = false;
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
        Debug.LogWarning($"[MQTT] Connection lost to {brokerAddress}:{brokerPort}");
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
    }
}
