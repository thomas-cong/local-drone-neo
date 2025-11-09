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

        if (centerPosition.HasValue)
        {
            Debug.Log($"[Voxel] Center voxel at ({centerPosition.Value.x:F2}, {centerPosition.Value.y:F2}, {centerPosition.Value.z:F2})");
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
            Debug.Log($"[Voxel] Axis test {name} at {position}");
        }
    }

    public static void Clear()
    {
        VoxelRendererManager.Instance.ClearAll();
    }
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

    private readonly List<GameObject> _pool = new List<GameObject>();
    private readonly List<GameObject> _active = new List<GameObject>();

    public void RenderCube(Vector3 position, float size, Color color)
    {
        var cube = GetCube();
        cube.transform.position = position;
        cube.transform.localScale = Vector3.one * size;
        var renderer = cube.GetComponent<MeshRenderer>();
        renderer.sharedMaterial.color = color;
        cube.SetActive(true);
        _active.Add(cube);
    }

    public void ClearAll()
    {
        foreach (var cube in _active)
        {
            cube.SetActive(false);
            _pool.Add(cube);
        }
        _active.Clear();
    }

    private GameObject GetCube()
    {
        if (_pool.Count > 0)
        {
            var cube = _pool[_pool.Count - 1];
            _pool.RemoveAt(_pool.Count - 1);
            return cube;
        }

        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Object.Destroy(go.GetComponent<Collider>());
        var renderer = go.GetComponent<MeshRenderer>();
        Shader shader = Shader.Find("Unlit/Color") ?? Shader.Find("Unlit/Texture") ?? Shader.Find("Standard");
        var material = new Material(shader);
        renderer.sharedMaterial = material;
        go.transform.SetParent(transform);
        return go;
    }
}
