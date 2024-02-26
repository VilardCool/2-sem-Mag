using System;
using System.Reflection;
using System.Threading;
using Parcs;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;

namespace CentroidLinkage
{
    class MainModule : IModule
    {
        public static void Main(string[] args)
        {

            var job = new Job();
            job.AddFile(Assembly.GetExecutingAssembly().Location);
            if (job == null)
            {
                Console.WriteLine("File doesn't exist");
                return;
            }

            (new MainModule()).Run(new ModuleInfo(job, null));
            Console.ReadKey();
        }
        
        static double EucDistance(List<double> a, List<double> b)
        {
            return Math.Sqrt(Math.Pow(a[0] - b[0], 2) + Math.Pow(a[1] - b[1], 2) + Math.Pow(a[2] - b[2], 2));
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

        public void Run(ModuleInfo info, CancellationToken token = default(CancellationToken))
        {
            int amount = 16;

            Console.Write("Number of colors: ");
            amount = Convert.ToInt32(Console.ReadLine());

            Console.Write("File name: ");
            string file = Console.ReadLine();

            Bitmap newImage = (Bitmap)Image.FromFile("input/" + file + ".png");

            Console.Write("Color divide: ");
            int divided = Convert.ToInt32(Console.ReadLine());

            Console.Write("Sort: ");
            int sort = Convert.ToInt32(Console.ReadLine());

            List<List<Tuple<double, double[]>>> dist = new List<List<Tuple<double, double[]>>>();

            List<Tuple<double, double[]>> distR = new List<Tuple<double, double[]>>();
            List<Tuple<double, double[]>> distG = new List<Tuple<double, double[]>>();
            List<Tuple<double, double[]>> distB = new List<Tuple<double, double[]>>();

            List<Tuple<double, double[]>> distOrig = new List<Tuple<double, double[]>>();

            List<List<double>> img = new List<List<double>>();

            for (int i = 0; i < newImage.Height; i++)
            {
                for (int j = 0; j < newImage.Width; j++)
                {
                    double r = newImage.GetPixel(j, i).R / 255.0;
                    double g = newImage.GetPixel(j, i).G / 255.0;
                    double b = newImage.GetPixel(j, i).B / 255.0;
                    double d = Math.Sqrt(r * r + g * g + b * b);

                    img.Add(new List<double>() { r, g, b });

                    double[] color = new double[3] { r, g, b };

                    Tuple<double, double[]> dis = new Tuple<double, double[]>(d, color);

                    if (divided == 1)
                    {
                        if (r >= g && r >= b) distR.Add(dis);
                        else if (g >= r && g >= b) distG.Add(dis);
                        else if (b >= r && b >= g) distB.Add(dis);
                    }
                    else distOrig.Add(dis);
                }
            }

            DateTime time = DateTime.Now;
            Console.WriteLine("Waiting for result...");

            if (divided == 1)
            {
                dist.Add(distR);
                dist.Add(distG);
                dist.Add(distB);
            }
            else dist.Add(distOrig);

            if (sort == 1)
                for (int k = 0; k < dist.Count; k++)
                {
                    var newDist = dist[k].OrderBy(c => c.Item1).ThenBy(b => b.Item2[0]).ThenBy(c => c.Item2[1]).ToList();
                    dist[k] = newDist;
                }

            List<List<double>> colors = new List<List<double>>();

            for (int i = 0; i < dist.Count; i++)
                for (int j = 0; j < dist[i].Count; j++)
                {
                    List<double> color = new List<double>();
                    for (int k = 0; k < 3; k++) color.Add(dist[i][j].Item2[k]);

                    if (sort == 1 && colors.Count > 0
                        && colors[colors.Count - 1][0] == color[0]
                        && colors[colors.Count - 1][1] == color[1]
                        && colors[colors.Count - 1][2] == color[2]) continue;

                    colors.Add(color);
                }

            const int pointsNum = 4;
            var points = new IPoint[pointsNum];
            var channels = new IChannel[pointsNum];
            for (int i = 0; i < pointsNum; ++i)
            {
                points[i] = info.CreatePoint();
                channels[i] = points[i].CreateChannel();
                points[i].ExecuteClass("CentroidLinkage.Module");
            }

            List<List<int>> res = new List<List<int>>();
            List<List<double>> result = new List<List<double>>();

            int part = colors.Count / pointsNum;
            int lastPart = part + colors.Count % pointsNum;
            int amountP = amount / pointsNum;
            int lastAmount = amountP + amount % pointsNum;

            for (int j = 0; j < pointsNum; j++)
            {
                if (j == pointsNum - 1)
                {
                    channels[j].WriteData(lastAmount);
                    channels[j].WriteObject(colors.GetRange(part * j, lastPart));
                }
                else
                {
                    channels[j].WriteData(amountP);
                    channels[j].WriteObject(colors.GetRange(part * j, part));
                }
            }

            for (int j = 0; j < pointsNum; j++)
            {
                List<List<double>> resu = (List<List<double>>)channels[j].ReadObject();
                for (int k = 0; k < resu.Count; k++) result.Add(resu[k]);
            }

            res = CentroidLinkage(img, result);

            Console.WriteLine("Result: {0}s", Math.Round((DateTime.Now - time).TotalSeconds, 3));

            var bitmap = new Bitmap(newImage.Width, newImage.Height);

            for (var y = 0; y < bitmap.Height; y++)
            {
                for (var x = 0; x < bitmap.Width; x++)
                {
                    Color color = Color.FromArgb(res[y * bitmap.Width + x][0], res[y * bitmap.Width + x][1], res[y * bitmap.Width + x][2]);
                    bitmap.SetPixel(x, y, color);
                }
            }

            string output = "output/" + bitmap.Width + "x" + bitmap.Height + "_" + amount + "_" + divided + sort + ".png";

            bitmap.Save(output);

            Console.Read();

        }
    }
}
