import java.io.FileWriter;
import java.io.IOException;

public class Logger {

    private static Logger instance;
    private FileWriter fileWriter;

    private Logger() {
        try {
            fileWriter = new FileWriter("log.txt", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Logger getInstance() {
        if (instance == null) {
            instance = new Logger();
        }
        return instance;
    }

    public void log(String message) {
        try {
            fileWriter.write(message + "\n");
            fileWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}


public class Logger {
    public Logger() {}

    public void log(String message) {
        System.out.println("Log: " + message);
    }
}

public class Main {
    public static void main(String[] args) {
        Logger logger1 = new Logger();
        logger1.log("This is the first logger.");

        Logger logger2 = new Logger();
        logger2.log("This is the second logger.");
    }
}