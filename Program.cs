using Microsoft.ML;
using System.IO;
using System;
using System.Threading.Channels;

namespace SystemPredictionsEats
{
    internal class Program
    {
        static void Main(string[] args)
        {
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
                //{11, "Buñuelo" },
                //{12, "Natilla" },
                //{13, "Pan de Bono" },
                //{14, "Garbanzos con callo"},
                //{15, "Hormiga culona" },
                //{16, "Gelatina de pata"},
                //{17, "Mojojoy" },
                //{18, "sopa de criadilla" },
                //{19, "Chanfaina" },
                //{20, "Pelanga" },
                //{21, "Salpicon"},
                //{22, "chunchullo"},
                //{23,  "Paella de Mariscos" }

            };

            var userRating = new List<EatRating>();

            Console.WriteLine("Califica las siguientes comidas de 1 a 10. Escribe 0 si no las has provado: ");

            foreach (var food in foods)
            {
                Console.WriteLine($"{food.Key}-{food.Value}: ");
                if(float.TryParse(Console.ReadLine(), out float rating) && rating > 0)
                {
                    userRating.Add(new EatRating
                    {
                        UserId = 1,
                        FoodId = food.Key,
                        label = rating
                    }); 
                }
            }

            var otherRathings = new List<EatRating>
            {
                new EatRating {UserId = 2, FoodId = 1, label = 9},
                new EatRating {UserId = 2, FoodId = 2, label = 7},
                new EatRating {UserId = 2, FoodId = 4, label = 8},
                new EatRating {UserId = 2, FoodId = 5, label = 6},
                new EatRating {UserId = 2, FoodId = 7, label = 4},
                //new EatRating {UserId = 3, FoodId = 13, label = 3},
                //new EatRating {UserId = 3, FoodId = 15, label = 9},
                //new EatRating {UserId = 3, FoodId = 18, label = 8},
                //new EatRating {UserId = 3, FoodId = 20, label = 10},
                //new EatRating {UserId = 3, FoodId = 21, label = 10},
                //new EatRating {UserId = 3, FoodId = 22, label = 7},
                //new EatRating {UserId = 3, FoodId = 23, label = 7},
                new EatRating {UserId = 4, FoodId = 3, label = 8},
                new EatRating {UserId = 4, FoodId = 6, label = 6},
                new EatRating {UserId = 4, FoodId = 10, label = 10},
                //new EatRating {UserId = 4, FoodId = 21, label = 7},
                new EatRating {UserId = 5, FoodId = 4, label = 7},
                new EatRating {UserId = 5, FoodId = 9, label = 3},
                //new EatRating {UserId = 5, FoodId = 12, label = 4},
                //new EatRating {UserId = 5, FoodId = 22, label = 10},
                new EatRating {UserId = 6, FoodId = 1, label = 10},
                //new EatRating {UserId = 6, FoodId = 12, label = 6},
                //new EatRating {UserId = 6, FoodId = 17, label = 4},
                //new EatRating {UserId = 6, FoodId = 20, label = 7},
            };

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
                Console.WriteLine("Ingrese ID de Usuario: "); ;
                float userID = float.Parse(Console.ReadLine());

                Console.WriteLine("Ingrese ID de la comida: ");
                float foodID = float.Parse(Console.ReadLine()); 

                var prediction = predictionsEngine.Predict(new EatRating
                {
                    UserId = userID,
                    FoodId = foodID,

                });

                Console.WriteLine($"Prediccion de clasificacion para el usuario {userID} en la comida {foods[(int)foodID]}: {prediction.Score:0.00}");

                //Todo
                // colores, y rellenar la matriz
            }
        }
    }
}
