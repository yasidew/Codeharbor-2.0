import json

# Patterns for Singleton types with placeholders
patterns = [
    {
        "type": "MultithreadedLazySingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "SerializationSafeSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.Serializable;\n\npublic class {name} implements Serializable {{\n\n    private static final long serialVersionUID = 1L;\n    private static final {name} INSTANCE = new {name}();\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return INSTANCE;\n    }}\n\n    protected Object readResolve() {{\n        return INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "EnumBasedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public enum {name} {{\n\n    INSTANCE;\n\n    public void someMethod() {{\n        System.out.println(\"Singleton with enum\");\n    }}\n\n}}"
    },
    {
        "type": "RetryFallbackSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            for (int retries = 0; retries < 3; retries++) {{\n                try {{\n                    instance = new {name}();\n                    break;\n                }} catch (Exception e) {{\n                    System.out.println(\"Retrying...\" + retries);\n                }}\n            }}\n            if (instance == null) {{\n                instance = new {name}(); // Fallback logic\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithCounter",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n    private static int counter = 0;\n\n    private {name}() {{\n        counter++;\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public int getInstanceCount() {{\n        return counter;\n    }}\n\n}}"
    },
    {
        "type": "MultiTenantSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static final Map<String, {name}> instances = new HashMap<>();\n\n    private {name}() {{}}\n\n    public static {name} getInstance(String tenantId) {{\n        synchronized (instances) {{\n            return instances.computeIfAbsent(tenantId, k -> new {name}());\n        }}\n    }}\n\n    public static int getTenantCount() {{\n        return instances.size();\n    }}\n\n}}"
    },
    {
        "type": "ReflectionProofSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final {name} instance = new {name}();\n\n    private {name}() {{\n        if (instance != null) {{\n            throw new IllegalStateException(\"Singleton instance already created\");\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LoggerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.FileWriter;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static volatile {name} instance;\n    private FileWriter writer;\n\n    private {name}() {{\n        try {{\n            writer = new FileWriter(\"application.log\", true);\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public synchronized void log(String message, String level) {{\n        try {{\n            writer.write(level + \": \" + message + \"\\n\");\n            writer.flush();\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public void close() {{\n        try {{\n            writer.close();\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n}}"
    },
    {
        "type": "DatabaseConnectionPoolSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.sql.Connection;\nimport java.sql.DriverManager;\nimport java.sql.SQLException;\nimport java.util.concurrent.ArrayBlockingQueue;\nimport java.util.concurrent.BlockingQueue;\n\npublic class {name} {{\n\n    private static final int POOL_SIZE = 5;\n    private static {name} instance;\n    private BlockingQueue<Connection> connectionPool;\n\n    private {name}() {{\n        connectionPool = new ArrayBlockingQueue<>(POOL_SIZE);\n        for (int i = 0; i < POOL_SIZE; i++) {{\n            try {{\n                connectionPool.add(DriverManager.getConnection(\"jdbc:mysql://localhost:3306/mydb\", \"user\", \"password\"));\n            }} catch (SQLException e) {{\n                e.printStackTrace();\n            }}\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public Connection getConnection() throws InterruptedException {{\n        return connectionPool.take();\n    }}\n\n    public void releaseConnection(Connection connection) {{\n        connectionPool.offer(connection);\n    }}\n\n}}"
    },
    {
        "type": "ConfigurationManagerSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.Properties;\nimport java.io.FileInputStream;\nimport java.io.IOException;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Properties properties;\n\n    private {name}() {{\n        properties = new Properties();\n        try (FileInputStream fis = new FileInputStream(\"application.properties\")) {{\n            properties.load(fis);\n        }} catch (IOException e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public String getProperty(String key) {{\n        return properties.getProperty(key);\n    }}\n\n}}"
    },
    {
        "type": "ServiceLocatorSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.HashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private Map<String, Object> services;\n\n    private {name}() {{\n        services = new HashMap<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public void registerService(String name, Object service) {{\n        services.put(name, service);\n    }}\n\n    public Object getService(String name) {{\n        return services.get(name);\n    }}\n\n}}"
    },
    {
        "type": "CacheSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.LinkedHashMap;\nimport java.util.Map;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private final int CACHE_SIZE = 5;\n    private Map<String, String> cache;\n\n    private {name}() {{\n        cache = new LinkedHashMap<String, String>(CACHE_SIZE, 0.75f, true) {{\n            protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {{\n                return size() > CACHE_SIZE;\n            }}\n        }};\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n    public void put(String key, String value) {{\n        cache.put(key, value);\n    }}\n\n    public String get(String key) {{\n        return cache.get(key);\n    }}\n\n}}"
    }
]

# Generate examples for each pattern
examples = []
for pattern in patterns:
    for i in range(5):  # Generate 5 variations per pattern
        class_name = f"{pattern['type']}Example{i + 1}"
        examples.append({
            "type": pattern["type"],
            "input": pattern["input_template"].format(name=class_name),
            "output": pattern["output_template"].format(name=class_name)
        })

# Save the dataset to a JSON file
dataset_filename = "augmented_singleton_data.json"
with open(dataset_filename, "w") as json_file:
    json.dump(examples, json_file, indent=4)

print(f"Dataset generated and saved to {dataset_filename}.")
