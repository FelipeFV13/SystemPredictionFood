using Microsoft.ML;
using System.IO;
using System;
using System.Threading.Channels;

namespace SystemPredictionsEats
{
    internal class Program
    {
        private static float userID;
        private static float foodID;

        static void Main(string[] args)
        {
            Random random = new Random();
            var context = new MLContext();

            var foods = new Dictionary<int, string>
            {
                {1, "Bandeja Paisa" },
                {2,"sancocho"},
                {3, "Changua" },
                {4, "Caldo de Costilla" },
                {5, "Lechona" },
                {6, "Ajiaco" },
                {7,"Tamal" },
                {8, "Empanadas" },
                {9, "Arepas" },
                {10, "Fritanga" },
                {11, "Buñuelo" },
                {12, "Natilla" },
                {13, "Pan de Bono" },
                {14, "Garbanzos con callo"},
                {15, "Hormiga culona" },
                {16, "Gelatina de pata"},
                {17, "Mojojoy" },
                {18, "sopa de criadilla" },
                {19, "Chanfaina" },
                {20, "Pelanga" },
                {21, "Salpicon"},
                {22, "chunchullo"},
                {23,  "Paella de Mariscos" }

            };

            var userRating = new List<EatRating>();

            Console.WriteLine("Califica las siguientes comidas de 1 a 10. Escribe 0 si no las has provado: ");

            foreach (var food in foods)
            {
                Console.WriteLine($"{food.Key}-{food.Value}: ");
                if (float.TryParse(Console.ReadLine(), out float rating) && rating > 0)
                {
                    userRating.Add(new EatRating
                    {
                        UserId = 1,
                        FoodId = food.Key,
                        label = rating
                    });
                }
            }


            var otherRathings = new List<EatRating>();

            // Se crea este bucle for anidado para llenar la lista de other Rathings
            // con 10 usuarios que califican los 23 platos de comida colombiana.
            // Los usuarios con id par van a calificar comida con el id impar
            // Los usuarios con id impar van a calificar comida con id par

            for (int i = 1; i < 10; i++)
            {
                for (int j = 1; j < 24; j++)
                {
                    int randomNumber = random.Next(1, 11);

                    if (i % 2 == 0)
                    {
                        if (j % 2 != 0)
                        {
                            otherRathings.Add(new EatRating { UserId = i, FoodId = j, label = randomNumber });
                        }

                    }
                    else
                    {
                        otherRathings.Add(new EatRating { UserId = i, FoodId = j, label = randomNumber });
                    }

                }
            }

            var allRatings = userRating.Concat(otherRathings);

            var data = context.Data.LoadFromEnumerable(allRatings);

            var option = new Microsoft.ML.Trainers.MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(EatRating.UserId),
                MatrixRowIndexColumnName = nameof(EatRating.FoodId),
                LabelColumnName = nameof(EatRating.label),
                NumberOfIterations = 20,
                ApproximationRank = 100,
            };


            var pipeline =
                context.Transforms.Conversion.MapValueToKey("UserId")
                .Append(context.Transforms.Conversion.MapValueToKey("FoodId"))
                .Append(context.Recommendation().Trainers.MatrixFactorization(option));

            var model = pipeline.Fit(data);
            var predictionsEngine = context.Model.CreatePredictionEngine<EatRating, FoodPrediction>(model);

            while (true)
            {
                Console.WriteLine("Si escribes un usuario con Id par, coloca un plato con id par para ver la predicción.");
                Console.WriteLine("Si escribes un usuario con Id Impar, coloca un plato con id impar para ver la predicción.");

                Console.WriteLine("Ingrese ID de Usuario: "); ;
                float userID = float.Parse(Console.ReadLine());

                Console.WriteLine("Ingrese ID de la comida: ");
                float foodID = float.Parse(Console.ReadLine());

                var prediction = predictionsEngine.Predict(new EatRating
                {
                    UserId = userID,
                    FoodId = foodID,

                });

                if (prediction.Score <= 5)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Prediccion de clasificacion para el usuario {userID} en la comida {foods[(int)foodID]}: {prediction.Score:0.00}");
                    Console.ForegroundColor = ConsoleColor.White;
                }
                else if (prediction.Score <= 7)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"Prediccion de clasificacion para el usuario {userID} en la comida {foods[(int)foodID]}: {prediction.Score:0.00}");
                    Console.ForegroundColor = ConsoleColor.White;
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"Prediccion de clasificacion para el usuario {userID} en la comida {foods[(int)foodID]}: {prediction.Score:0.00}");
                    Console.ForegroundColor = ConsoleColor.White;
                };
            }

        }
    }
 }

