if (args.Length == 0)
{
    RunDefaultExamples();
    return;
}

switch (args[0])
{
    case "help":
    case "--help":
    case "-h":
        PrintHelp();
        break;
    case "--dump-state":
        if (args.Length > 2)
        {
            throw new ArgumentException("Usage: --dump-state [output-directory]");
        }

        DumpState.Run(args.Length == 2 ? args[1] : "state_output");
        break;
    case "encode-hash":
        EncodeHashExample.Run(args[1..]);
        break;
    case "kernel-xtest":
        KernelParityExample.Run(args[1..]);
        break;
    case "insert-bench":
        InsertBenchmarkExample.Run(args[1..]);
        break;
    case "sync-bench":
        SyncBenchmarkExample.Run(args[1..]);
        break;
    case "stream-bw":
        StreamBandwidthExample.Run(args[1..]);
        break;
    default:
        throw new ArgumentException($"Unknown example command '{args[0]}'. Use 'help' to list commands.");
}

static void RunDefaultExamples()
{
    Console.WriteLine("=== Basic Usage ===");
    BasicUsage.Run();

    Console.WriteLine("\n=== IdMap Example ===");
    IdMapExample.Run();
}

static void PrintHelp()
{
    Console.WriteLine("TurboVector examples");
    Console.WriteLine("  (no arguments)       Run BasicUsage and IdMapExample");
    Console.WriteLine("  encode-hash          Print deterministic encode-stage fingerprints");
    Console.WriteLine("  kernel-xtest         Compare deterministic top-K result sets across platforms");
    Console.WriteLine("  insert-bench         Measure bulk/warm/single adds and removals");
    Console.WriteLine("  sync-bench           Measure full, append, and removal sync latency");
    Console.WriteLine("  stream-bw            Measure sequential memory-read bandwidth");
    Console.WriteLine("  --dump-state [path]  Write state files for inspection");
}
