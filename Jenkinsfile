#!groovy​

// All the types of build we'll ideally run if suitable nodes exist
def desiredBuilds = [
    ["cuda9", "linux", "x86_64", "python27"] as Set,
    ["cuda8", "linux", "x86_64", "python27"] as Set,
    ["cpu_only", "linux", "x86_64", "python27"] as Set,
    ["cuda9", "linux", "x86", "python27"] as Set,
    ["cuda8", "linux", "x86", "python27"] as Set,
    ["cpu_only", "linux", "x86", "python27"] as Set,
    ["cuda9", "mac", "python27"] as Set,
    ["cuda8", "mac", "python27"] as Set,
    ["cpu_only", "mac", "python27"] as Set,
    ["cuda9", "linux", "x86_64", "python3"] as Set,
    ["cuda8", "linux", "x86_64", "python3"] as Set,
    ["cpu_only", "linux", "x86_64", "python3"] as Set,
    ["cuda9", "linux", "x86", "python3"] as Set,
    ["cuda8", "linux", "x86", "python3"] as Set,
    ["cpu_only", "linux", "x86", "python3"] as Set,
    ["cuda9", "mac", "python3"] as Set,
    ["cuda8", "mac", "python3"] as Set,
    ["cpu_only", "mac", "python3"] as Set] 

//--------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------
// Wrapper around setting of GitHUb commit status curtesy of https://groups.google.com/forum/#!topic/jenkinsci-issues/p-UFjxKkXRI
// **NOTE** since that forum post, stage now takes a Closure as the last argument hence slight modification 
void buildStep(String message, Closure closure) {
    stage(message)
    {
        try {
            setBuildStatus(message, "PENDING");
            closure();
        } catch (Exception e) {
            setBuildStatus(message, "FAILURE");
        }
    }
}

void setBuildStatus(String message, String state) {
    // **NOTE** ManuallyEnteredCommitContextSource set to match the value used by bits of Jenkins outside pipeline control
    step([
        $class: "GitHubCommitStatusSetter",
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/genn-team/genn/"],
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "continuous-integration/jenkins/branch"],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}

//--------------------------------------------------------------------------
// Entry point
//--------------------------------------------------------------------------
// Build dictionary of available nodes and their labels
def availableNodes = [:]
for(node in jenkins.model.Jenkins.instance.nodes) {
    if(node.getComputer().isOnline() && node.getComputer().countIdle() > 0) {
        availableNodes[node.name] = node.getLabelString().split() as Set
    }
}

// Add master if it has any idle executors
if(jenkins.model.Jenkins.instance.toComputer().countIdle() > 0) {
    availableNodes["master"] = jenkins.model.Jenkins.instance.getLabelString().split() as Set
}

// Loop through the desired builds
def builderNodes = []
for(b in desiredBuilds) {
    // Loop through all available nodes
    for(n in availableNodes) {
        // If, after subtracting this node's labels, all build properties are satisfied
        if((b - n.value).size() == 0) {
            print "${n.key} -> ${b}";
            
            // Add node's name to list of builders and remove it from dictionary of available nodes
            // **YUCK** for some reason tuples aren't serializable so need to add an arraylist
            builderNodes.add([n.key, n.value])
            availableNodes.remove(n.key)
            break
        }
    }
}

//--------------------------------------------------------------------------
// Parallel build step
//--------------------------------------------------------------------------
// **YUCK** need to do a C style loop here - probably due to JENKINS-27421 
def builders = [:]
for(b = 0; b < builderNodes.size; b++) {
    // **YUCK** meed to bind the label variable before the closure - can't do 'for (label in labels)'
    def nodeName = builderNodes.get(b).get(0)
    def nodeLabel = builderNodes.get(b).get(1)
   
    // Create a map to pass in to the 'parallel' step so we can fire all the builds at once
    builders[nodeName] = {
        node(nodeName) {
            def installationStageName =  "Installation (" + env.NODE_NAME + ")";
 
            // Customise this nodes environment so GeNN environment variable is set and genn binaries are in path
            // **NOTE** these are NOT set directly using env.PATH as this makes the change across ALL nodes which means you get a randomly mangled path depending on node startup order
            withEnv(["GENN_PATH=" + pwd() + "/genn",
                     "PATH+GENN=" + pwd() + "/genn/lib/bin"]) {
                stage(installationStageName) {
                    echo "Checking out GeNN";

                    // Deleting existing checked out version of pynn_genn
                    sh "rm -rf pynn_genn";

                    dir("pynn_genn") {
                        // Checkout pynn_genn here
                        // **NOTE** because we're using multi-branch project URL is substituted here
                        checkout scm
                    }

                    // **NOTE** only try and set build status AFTER checkout
                    try {
                        setBuildStatus(installationStageName, "PENDING");

                        // If GeNN exists
                        if(fileExists("genn")) {
                            echo "Updating GeNN";
                            
                            // Pull from repository
                            dir("genn") {
                                sh "git pull";
                            }
                        }
                        else {
                            echo "Cloning GeNN";
                            sh "git clone -b python_wrapper https://github.com/genn-team/genn.git";
                        }

                        // Remove existing virtualenv
                        sh "rm -rf virtualenv";

                        // Create new one
                        echo "Creating virtualenv";
                        sh "virtualenv --system-site-packages virtualenv";

                    } catch (Exception e) {
                        setBuildStatus(installationStageName, "FAILURE");
                    }
                }

                buildStep("Installing Python modules(" + env.NODE_NAME + ")") {
                    // Activate virtualenv
                    dir("virtualenv") {
                        sh ". bin/activate";
                    }
                    
                    // Install packages
                    sh "pip install neo pynn quantities nose nose_testconfig";
                }

                buildStep("Building PyGeNN (" + env.NODE_NAME + ")") {
                    dir("genn") {
                        // Build dynamic LibGeNN
                        // **TODO** only do this stage if anything's changed in GeNN
                        echo "Building LibGeNN";
                        if("cpu_only" in nodeLabel) {
                            sh "make -f lib/GNUMakefileLibGeNN DYNAMIC=1 CPU_ONLY=1 LIBGENN_PATH=pygenn/genn_wrapper/";

                            // If node is a mac, re-label library
                            if("mac" in nodeLabel) {
                                sh "install_name_tool -id \"@loader_path/libgenn_CPU_ONLY_DYNAMIC.dylib\" pygenn/genn_wrapper/libgenn_CPU_ONLY_DYNAMIC.dylib";
                            }
                        }
                        else {
                            sh "make -f lib/GNUMakefileLibGeNN DYNAMIC=1 LIBGENN_PATH=pygenn/genn_wrapper/";

                            // If node is a mac, re-label library
                            if("mac" in nodeLabel) {
                                sh "install_name_tool -id \"@loader_path/libgenn_DYNAMIC.dylib\" pygenn/genn_wrapper/libgenn_DYNAMIC.dylib";
                            }
                        }
                        
                        // Activate virtualenv
                        dir("virtualenv") {
                            sh ". bin/activate";
                        }
                        echo "Building Python module";
                        sh "python setup.py install"
                    }
                }

                buildStep("Running tests (" + env.NODE_NAME + ")") {
                    // Activate virtualenv
                    dir("virtualenv") {
                        sh ". bin/activate";
                    }
                    
                    dir("pynn_genn/test/system") {
                        // Generate unique name for message
                        def uniqueMsg = "msg_" + env.NODE_NAME;
                        
                        // Run tests, piping output to message file
                        sh "nosetests -s --with-xunit test_genn.py 1> \"" + uniqueMsg + "\" 2> \"" + uniqueMsg + "\"";
                        
                        // Archive output
                        archive uniqueMsg;
                    }
                }

                buildStep("Gathering test results (" + env.NODE_NAME + ")") {
                    // Process JUnit test output
                    junit "pynn_genn/test/**/nosetests.xml";
                }
            }
        }
    }
}

// Run builds in parallel
parallel builders