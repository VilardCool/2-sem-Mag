using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading;
using Parcs;

namespace CentroidLinkage
{
    public class Module : IModule
    {
        static double EucDistance(List<double> a, List<double> b)
        {
            return Math.Sqrt(Math.Pow(a[0] - b[0], 2) + Math.Pow(a[1] - b[1], 2) + Math.Pow(a[2] - b[2], 2));
        }

        static List<int> GetMin(List<List<double>> mat)
        {
            double min = double.MaxValue;
            int a = 0;
            int b = 1;
            for (int i = 0; i < mat.Count; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    if (mat[i][j] < min)
                    {
                        min = mat[i][j];
                        a = Math.Min(i, j);
                        b = Math.Max(i, j);
                    }
                }
            }

            return new List<int>() { a, b };
        }


        static void ChangeCentroid(ref List<List<double>> mat, ref List<List<double>> colors, int a, int b)
        {
            List<double> centroid = new List<double>() { 0, 0, 0 };

            for (int i = 0; i < 3; i++)
            {
                centroid[i] = (colors[a][i] + colors[b][i]) / 2.0;
            }

            for (int i = a + 1; i < colors.Count; i++)
            {
                if (i > b) mat[i].RemoveAt(b);
                mat[i].RemoveAt(a);
            }

            mat.RemoveAt(b);
            mat.RemoveAt(a);
            colors.RemoveAt(b);
            colors.RemoveAt(a);

            colors.Add(centroid);

            List<double> row = new List<double>();

            for (int i = 0; i < colors.Count; i++)
            {
                row.Add(EucDistance(colors[i], centroid));
            }
            mat.Add(row);
        }

        static List<int> FindClosestCentroids(List<List<double>> img, List<List<double>> colors)
        {
            int m = img.Count;
            List<int> centroids = new List<int>();

            for (int i = 0; i < m; i++)
            {
                List<double> distances = new List<double>();

                for (int j = 0; j < colors.Count; j++)
                {
                    distances.Add(EucDistance(img[i], colors[j]));
                }

                centroids.Add(distances.IndexOf(distances.Min()));
            }

            return centroids;
        }

        static List<List<int>> CentroidLinkage(List<List<double>> img, List<List<double>> colors)
        {
            List<int> idx = FindClosestCentroids(img, colors);

            List<List<int>> recolored_img = new List<List<int>>();

            for (int i = 0; i < idx.Count; i++)
            {
                for (int j = 0; j < colors.Count; j++)
                {
                    if (idx[i] == j)
                    {
                        List<int> color = new List<int>();
                        for (int k = 0; k < 3; k++)
                        {
                            color.Add((int)(colors[j][k] * 255));
                        }
                        recolored_img.Add(color);
                        break;
                    }
                }
            }

            return recolored_img;
        }

        private static List<List<double>> Linkage(int amount, List<List<double>> colors)
        {
            List<List<double>> mat = new List<List<double>>();

            for (int i = 0; i < colors.Count; i++)
            {
                List<double> row = new List<double>();

                for (int j = 0; j < i; j++)
                {
                    row.Add(EucDistance(colors[i], colors[j]));
                }

                mat.Add(row);
            }

            while (mat.Count > amount)
            {
                List<int> ab = GetMin(mat);
                int fir = ab[0];
                int sec = ab[1];

                ChangeCentroid(ref mat, ref colors, fir, sec);
            }

            return colors;
        }

        static List<List<double>> Reduce(int amountFin, List<List<double>> colors)
        {
            List<List<double>> result = new List<List<double>>();
            int amount = amountFin;
            int box = amount + amount / 2 + 1;
            /*
            if (amount * colors.Count / box <= amountFin * box)
            {
                amount = amountFin;
                box = amount + amount / 2 + 1;
            }
            */
            for (int i = 0; i <= colors.Count / box; i++)
            {
                int part = box;
                if (colors.Count - box * i < box)
                {
                    part = colors.Count - box * i;
                }

                List<List<double>> resu = Linkage(amount, colors.GetRange(box*i, part));
                for (int k = 0; k < resu.Count; k++) result.Add(resu[k]);

            }

            if (result.Count > amountFin) result = Reduce(amountFin, result);

            return result;
        }

        public void Run(ModuleInfo info, CancellationToken token = default(CancellationToken))
        {
            while (true)
            {
                int amount = info.Parent.ReadInt();

                List<List<double>> colors = (List<List<double>>)info.Parent.ReadObject();

                List<List<double>> result = Reduce(amount, colors);

                info.Parent.WriteObject(result);

            }
        }
    }
}