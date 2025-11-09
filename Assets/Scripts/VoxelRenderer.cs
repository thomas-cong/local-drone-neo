using System.Collections.Generic;
using UnityEngine;

public static class VoxelRenderer
{
    private const float DefaultHeightOffset = 0.2f;

    public static void RenderVoxel(Vector3 position, float size, Color color)
    {
        VoxelRendererManager.Instance.RenderCube(position, size, color);
    }

    public static void RenderMatrix(float[,] values, float spacing, float heightScale, float voxelSize, Color color, Vector3 anchor, float heightOffset = DefaultHeightOffset)
    {
        if (values == null)
        {
            return;
        }

        int rows = values.GetLength(0);
        int cols = values.GetLength(1);
        float centerRow = (rows - 1) * 0.5f;
        float centerCol = (cols - 1) * 0.5f;
        Vector3? centerPosition = null;

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                float height = values[r, c] * heightScale + heightOffset;
                var position = anchor + new Vector3((r - centerRow) * spacing, height, (c - centerCol) * spacing);
                VoxelRendererManager.Instance.RenderCube(position, voxelSize, color);
                if (Mathf.Abs(r - centerRow) < 0.5f && Mathf.Abs(c - centerCol) < 0.5f)
                {
                    centerPosition = position;
                }
            }
        }

    }

    public static void RenderAxisTest(Vector3 anchor)
    {
        var tests = new (Vector3 offset, Color color, string name)[]
        {
            (new Vector3(0f, 0f, 0.3f), Color.red, "Forward"),
            (new Vector3(0f, 0f, -0.3f), Color.blue, "Backward"),
            (new Vector3(0.3f, 0f, 0f), Color.green, "Right"),
            (new Vector3(-0.3f, 0f, 0f), Color.magenta, "Left"),
            (new Vector3(0f, 0.3f, 0f), Color.cyan, "Up"),
            (new Vector3(0f, -0.3f, 0f), Color.yellow, "Down")
        };

        foreach (var (offset, color, name) in tests)
        {
            var position = anchor + offset + Vector3.up * 0.1f;
            VoxelRendererManager.Instance.RenderCube(position, 0.03f, color);
        }
    }

    public static void Clear()
    {
        VoxelRendererManager.Instance.ClearAll();
    }

    public static void ApplyFrame(IEnumerable<VoxelData> voxels, bool useTemporalBlend, float blendFactor)
    {
        VoxelRendererManager.Instance.ApplyFrame(voxels, useTemporalBlend, blendFactor);
    }

    public static void RenderText(Vector3 position, string message, Color color)
    {
        VoxelRendererManager.Instance.RenderTextBillboard(position, message, color);
    }
}

public struct VoxelData
{
    public Vector3 Position;
    public float Size;
    public Color Color;
    public bool UseBillboard;
    public float BillboardScale;
    public Texture2D BillboardTexture;
}

internal enum VoxelPooledType
{
    Cube,
    Billboard,
    Text
}

internal sealed class VoxelRendererManager : MonoBehaviour
{
    private static VoxelRendererManager _instance;
    public static VoxelRendererManager Instance
    {
        get
        {
            if (_instance == null)
            {
                var go = new GameObject("VoxelRendererManager");
                DontDestroyOnLoad(go);
                _instance = go.AddComponent<VoxelRendererManager>();
            }
            return _instance;
        }
    }

    private readonly Dictionary<VoxelPooledType, Stack<GameObject>> _pools = new Dictionary<VoxelPooledType, Stack<GameObject>>();
    private readonly List<GameObject> _activeObjects = new List<GameObject>();
    private readonly List<VoxelData> _previousData = new List<VoxelData>();
    private readonly List<VoxelData> _currentData = new List<VoxelData>();

    public void RenderCube(Vector3 position, float size, Color color)
    {
        var cube = GetObject(VoxelPooledType.Cube);
        cube.transform.position = position;
        cube.transform.localScale = Vector3.one * size;
        cube.GetComponent<MeshRenderer>().sharedMaterial.color = color == default ? Color.red : color;
        _activeObjects.Add(cube);
    }

    public void ClearAll()
    {
        RecycleActiveObjects();
        _previousData.Clear();
    }

    private GameObject GetObject(VoxelPooledType type)
    {
        if (!_pools.TryGetValue(type, out var pool))
        {
            pool = new Stack<GameObject>();
            _pools[type] = pool;
        }

        if (pool.Count > 0)
        {
            var obj = pool.Pop();
            obj.SetActive(true);
            return obj;
        }

        GameObject created;
        switch (type)
        {
            case VoxelPooledType.Billboard:
                created = GameObject.CreatePrimitive(PrimitiveType.Quad);
                Object.Destroy(created.GetComponent<Collider>());
                break;
            case VoxelPooledType.Text:
                created = new GameObject("VoxelText");
                var textMesh = created.AddComponent<TextMesh>();
                textMesh.text = string.Empty;
                textMesh.alignment = TextAlignment.Center;
                textMesh.anchor = TextAnchor.MiddleCenter;
                textMesh.fontSize = 64;
                textMesh.color = Color.white;
                created.AddComponent<MeshRenderer>();
                break;
            default:
                created = GameObject.CreatePrimitive(PrimitiveType.Cube);
                Object.Destroy(created.GetComponent<Collider>());
                break;
        }

        created.AddComponent<VoxelPooledMarker>().Type = type;
        created.transform.SetParent(transform);
        return created;
    }

    public void ApplyFrame(IEnumerable<VoxelData> voxels, bool useTemporalBlend, float blendFactor)
    {
        RecycleActiveObjects();

        _currentData.Clear();
        _currentData.AddRange(voxels);
        bool canBlend = useTemporalBlend && blendFactor > 0f && blendFactor < 1f && _previousData.Count == _currentData.Count && _previousData.Count > 0;

        for (int i = 0; i < _currentData.Count; i++)
        {
            var data = _currentData[i];
            if (canBlend)
            {
                var prev = _previousData[i];
                data.Position = Vector3.Lerp(prev.Position, data.Position, blendFactor);
                data.Size = Mathf.Lerp(prev.Size, data.Size, blendFactor);
            }

            if (data.UseBillboard)
            {
                RenderBillboard(data.Position, data.BillboardScale, data.Color, data.BillboardTexture);
            }
            else
            {
                RenderCube(data.Position, data.Size, data.Color);
            }

            _currentData[i] = data;
        }

        _previousData.Clear();
        _previousData.AddRange(_currentData);
    }

    private void RenderBillboard(Vector3 position, float scale, Color color, Texture2D texture)
    {
        var quad = GetObject(VoxelPooledType.Billboard);
        quad.transform.position = position;
        quad.transform.localScale = Vector3.one * scale;
        var cam = Camera.main;
        if (cam != null)
        {
            quad.transform.LookAt(cam.transform);
            quad.transform.Rotate(0f, 180f, 0f);
        }
        var renderer = quad.GetComponent<MeshRenderer>();
        Shader shader = Shader.Find("Unlit/Texture") ?? Shader.Find("Unlit/Color");
        var material = new Material(shader);
        if (texture != null && material.HasProperty("_MainTex"))
        {
            material.SetTexture("_MainTex", texture);
            material.SetColor("_Color", color);
        }
        else
        {
            material.color = color;
        }
        renderer.sharedMaterial = material;
        _activeObjects.Add(quad);
    }

    public void RenderTextBillboard(Vector3 position, string message, Color color)
    {
        var textObj = GetObject(VoxelPooledType.Text);
        var textMesh = textObj.GetComponent<TextMesh>();
        textMesh.text = message;
        textMesh.color = color;
        textMesh.alignment = TextAlignment.Center;
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.characterSize = 0.1f;
        textMesh.fontSize = 64;
        textObj.transform.position = position;
        textObj.transform.localScale = Vector3.one * 0.05f;
        var cam = Camera.main;
        if (cam != null)
        {
            textObj.transform.LookAt(cam.transform);
            textObj.transform.Rotate(0f, 180f, 0f);
        }
        _activeObjects.Add(textObj);
    }

    private void RecycleActiveObjects()
    {
        foreach (var obj in _activeObjects)
        {
            obj.SetActive(false);
            var marker = obj.GetComponent<VoxelPooledMarker>();
            if (!_pools.TryGetValue(marker.Type, out var stack))
            {
                stack = new Stack<GameObject>();
                _pools[marker.Type] = stack;
            }
            stack.Push(obj);
        }
        _activeObjects.Clear();
    }
}

internal class VoxelPooledMarker : MonoBehaviour
{
    public VoxelPooledType Type;
}
