// Run examples
Console.WriteLine("=== Basic Usage ===");
BasicUsage.Run();

Console.WriteLine("\n=== IdMap Example ===");
IdMapExample.Run();

if (args.Contains("--dump-state"))
{
    Console.WriteLine("\n=== Dump State ===");
    DumpState.Run(args.Length > 1 ? args[1] : "state_output");
}
