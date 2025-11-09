using System;

public static class MatrixUtility
{
    public static void AddGaussianNoiseInPlace(float[,] matrix, float mean = 0f, float standardDeviation = 1f, int? seed = null)
    {
        ValidateMatrix(matrix);
        if (standardDeviation < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(standardDeviation), "Standard deviation must be non-negative.");
        }

        var random = seed.HasValue ? new Random(seed.Value) : RandomInstance.Instance;
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                float noise = standardDeviation <= 0f ? 0f : SampleGaussian(random, mean, standardDeviation);
                matrix[r, c] += noise;
            }
        }
    }

    public static int[,] ToBinaryMask(float[,] matrix, int[,] destination = null)
    {
        ValidateMatrix(matrix);
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        destination ??= new int[rows, cols];
        ValidateMask(destination, rows, cols);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                destination[r, c] = matrix[r, c] < 0f ? 0 : 1;
            }
        }

        return destination;
    }

    public static void ApplyBinaryMaskInPlace(float[,] matrix, int[,] mask)
    {
        ValidateMatrix(matrix);
        ValidateMask(mask, matrix.GetLength(0), matrix.GetLength(1));

        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (mask[r, c] == 0)
                {
                    matrix[r, c] = 0f;
                }
            }
        }
    }

    public static void MultiplyInPlace(float[,] matrix, float scalar)
    {
        ValidateMatrix(matrix);
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                matrix[r, c] *= scalar;
            }
        }
    }

    private static void ValidateMatrix(float[,] matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix));
        }
    }

    private static void ValidateMask(int[,] mask, int expectedRows, int expectedCols)
    {
        if (mask == null)
        {
            throw new ArgumentNullException(nameof(mask));
        }

        if (mask.GetLength(0) != expectedRows || mask.GetLength(1) != expectedCols)
        {
            throw new ArgumentException("Mask dimensions must match the matrix dimensions.", nameof(mask));
        }
    }

    private static float SampleGaussian(Random random, float mean, float standardDeviation)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return (float)(mean + standardDeviation * randStdNormal);
    }

    private static class RandomInstance
    {
        internal static readonly Random Instance = new Random();
    }
}
