import java.io.*;

class CallPythonForModelInfo {
    public static void main(String[] args) {
        try {
            String pythonExecutable = "python"; // or "python3" on some systems
            String pythonScriptPath = "./src/lab/lab1/final.py";

            // Start timer
            long startTime = System.currentTimeMillis();

            ProcessBuilder builder = new ProcessBuilder(pythonExecutable, pythonScriptPath);
            Process process = builder.start();

            BufferedReader stdInput = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String s;
            String bestModelName = "";
            String bestModelRocAucScore = "";
            while ((s = stdInput.readLine()) != null) {
                if (s.startsWith("BestModel:")) {
                    bestModelName = s.substring(s.indexOf(":") + 1).trim();
                } else if (s.startsWith("F1 Score:")) {
                    bestModelRocAucScore = s.substring(s.indexOf(":") + 1).trim();
                }
            }

            // Stop timer
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;

            System.out.println("The best model is: " + bestModelName + " with a F1 Score of: " + bestModelRocAucScore);
            System.out.println("Python script execution time: " + duration + " milliseconds");

            int exitVal = process.waitFor();
            if (exitVal != 0) {
                System.out.println("Python script exited with error code " + exitVal);
                BufferedReader stdError = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                while ((s = stdError.readLine()) != null) {
                    System.out.println("Error: " + s);
                }
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
