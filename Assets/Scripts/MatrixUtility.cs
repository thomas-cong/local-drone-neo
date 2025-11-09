using System;
using UnityEngine;

public static class MatrixUtility
{
    public static void AddGaussianNoiseInPlace(float[,] matrix, float mean = 0f, float standardDeviation = 1f, int? seed = null)
    {
        ValidateMatrix(matrix);
        if (standardDeviation < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(standardDeviation), "Standard deviation must be non-negative.");
        }

        var random = seed.HasValue ? new System.Random(seed.Value) : RandomInstance.Instance;
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

    public static void BoxBlurInPlace(float[,] matrix, int kernelRadius)
    {
        ValidateMatrix(matrix);
        if (kernelRadius <= 0)
        {
            return;
        }

        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var temp = new float[rows, cols];

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                float sum = 0f;
                int count = 0;
                for (int dr = -kernelRadius; dr <= kernelRadius; dr++)
                {
                    int rr = r + dr;
                    if (rr < 0 || rr >= rows)
                    {
                        continue;
                    }

                    for (int dc = -kernelRadius; dc <= kernelRadius; dc++)
                    {
                        int cc = c + dc;
                        if (cc < 0 || cc >= cols)
                        {
                            continue;
                        }

                        sum += matrix[rr, cc];
                        count++;
                    }
                }

                temp[r, c] = count > 0 ? sum / count : matrix[r, c];
            }
        }

        Array.Copy(temp, matrix, rows * cols);
    }

    public static float[,] UpscaleMatrix(float[,] source, int scale)
    {
        ValidateMatrix(source);
        if (scale <= 1)
        {
            return source;
        }

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        int newRows = rows * scale;
        int newCols = cols * scale;
        var result = new float[newRows, newCols];

        for (int r = 0; r < newRows; r++)
        {
            float srcRow = (float)r / scale;
            int r0 = Mathf.Clamp((int)Math.Floor(srcRow), 0, rows - 1);
            int r1 = Mathf.Clamp(r0 + 1, 0, rows - 1);
            float tRow = srcRow - r0;

            for (int c = 0; c < newCols; c++)
            {
                float srcCol = (float)c / scale;
                int c0 = Mathf.Clamp((int)Math.Floor(srcCol), 0, cols - 1);
                int c1 = Mathf.Clamp(c0 + 1, 0, cols - 1);
                float tCol = srcCol - c0;

                float top = Mathf.Lerp(source[r0, c0], source[r0, c1], tCol);
                float bottom = Mathf.Lerp(source[r1, c0], source[r1, c1], tCol);
                result[r, c] = Mathf.Lerp(top, bottom, tRow);
            }
        }

        return result;
    }

    public static int[,] UpscaleMask(int[,] source, int scale)
    {
        ValidateMask(source, source.GetLength(0), source.GetLength(1));
        if (scale <= 1)
        {
            return source;
        }

        int rows = source.GetLength(0);
        int cols = source.GetLength(1);
        int newRows = rows * scale;
        int newCols = cols * scale;
        var result = new int[newRows, newCols];

        for (int r = 0; r < newRows; r++)
        {
            int srcRow = Mathf.Clamp(r / scale, 0, rows - 1);
            for (int c = 0; c < newCols; c++)
            {
                int srcCol = Mathf.Clamp(c / scale, 0, cols - 1);
                result[r, c] = source[srcRow, srcCol];
            }
        }

        return result;
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

    private static float SampleGaussian(System.Random random, float mean, float standardDeviation)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return (float)(mean + standardDeviation * randStdNormal);
    }

    private static class RandomInstance
    {
        internal static readonly System.Random Instance = new System.Random();
    }
}
