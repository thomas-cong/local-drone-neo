using UnityEngine;

[CreateAssetMenu(fileName = "VoxelRenderSettings", menuName = "WallHack/Voxel Render Settings")]
public class VoxelRenderSettings : ScriptableObject
{
    [Header("Visualization Mode")]
    public bool condensedView = true;
    public bool showTestGrid = true;

    [Header("Condensed Parameters")]
    public float condensedSpacing = 0.01f;
    public float condensedHeightScale = 0.002f;
    public float condensedVoxelSize = 0.01f;

    [Header("Temporal Smoothing")]
    public bool enableTemporalBlend = true;
    [Range(0f, 1f)] public float blendFactor = 0.5f;

    public RenderingMode renderingMode = RenderingMode.Voxels;
    public bool blurMaskBeforeRender = false;
    public Texture2D billboardTexture;
    public float billboardScale = 0.03f;
    [Range(1, 8)] public int renderMatrixUpscale = 1;
    public float depthScale = 0.005f;
}

public enum RenderingMode
{
    Voxels,
    Billboards
}
