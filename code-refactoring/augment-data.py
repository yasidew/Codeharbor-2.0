import json

# Patterns for Singleton types with placeholders
patterns = [
    {
        "type": "EagerlyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static final {name} instance = new {name}();\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "LazyInitializedSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ThreadSafeSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static synchronized {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "BillPughSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private {name}() {{}}\n\n    private static class SingletonHelper {{\n        private static final {name} INSTANCE = new {name}();\n    }}\n\n    public static {name} getInstance() {{\n        return SingletonHelper.INSTANCE;\n    }}\n\n}}"
    },
    {
        "type": "SerializableSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.io.Serializable;\n\npublic class {name} implements Serializable {{\n\n    private static final long serialVersionUID = 1L;\n    private static {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    protected Object readResolve() {{\n        return getInstance();\n    }}\n\n}}"
    },
    {
        "type": "DoubleCheckedLockingSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static volatile {name} instance;\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            synchronized ({name}.class) {{\n                if (instance == null) {{\n                    instance = new {name}();\n                }}\n            }}\n        }}\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "StaticBlockSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static {name} instance;\n\n    static {{\n        try {{\n            instance = new {name}();\n        }} catch (Exception e) {{\n            e.printStackTrace();\n        }}\n    }}\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return instance;\n    }}\n\n}}"
    },
    {
        "type": "ObserverPatternSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "import java.util.ArrayList;\nimport java.util.List;\n\npublic class {name} {{\n\n    private static {name} instance;\n    private List<Observer> observers;\n\n    private {name}() {{\n        observers = new ArrayList<>();\n    }}\n\n    public static {name} getInstance() {{\n        if (instance == null) {{\n            instance = new {name}();\n        }}\n        return instance;\n    }}\n\n    public void addObserver(Observer observer) {{\n        observers.add(observer);\n    }}\n\n    public void notifyObservers() {{\n        for (Observer observer : observers) {{\n            observer.update();\n        }}\n    }}\n\n}}\n\ninterface Observer {{\n    void update();\n}}"
    },
    {
        "type": "ThreadLocalSingleton",
        "input_template": "public class {name} {{\n\n    public {name}() {{}}\n\n}}",
        "output_template": "public class {name} {{\n\n    private static final ThreadLocal<{name}> threadLocalInstance = ThreadLocal.withInitial({name}::new);\n\n    private {name}() {{}}\n\n    public static {name} getInstance() {{\n        return threadLocalInstance.get();\n    }}\n\n}}"
    },
    {
        "type": "SingletonWithFallbackInstance",
        "input_template": "public class {name} {\n\n    public {name}() {}\n\n}",
        "output_template": "public class {name} {\n\n    private static {name} instance;\n\n    private {name}() {}\n\n    public static {name} getInstance() {\n        if (instance == null) {\n            try {\n                instance = new {name}();\n            } catch (Exception e) {\n                instance = new {name}(); // Fallback logic\n            }\n        }\n        return instance;\n    }\n\n}"
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
