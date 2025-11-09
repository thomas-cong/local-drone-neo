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
}
