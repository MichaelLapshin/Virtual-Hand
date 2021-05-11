using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Net.Sockets;
using Debug = UnityEngine.Debug;
using System.Windows.Input;

public class HandController : MonoBehaviour
{
    // Game variables
    private UnityEngine.GameObject[] movableLimbs;
    private UnityEngine.Rigidbody[] rigidBodies;
    private float[] limb_velocities;

    // Process variables
    private ClientConnectionHandler connection;
    private Process process = null;
    private Thread thread = null;
    private bool running = false;
    private object limb_velocities_locker = new object();

    // Start is called before the first frame update
    void Start()
    {
        // Object fetching related logic
        while (GeneralData.isReady() == false)
        {
        }

        movableLimbs = GeneralData.getHingeLimbs("Hand Real");
        rigidBodies = new Rigidbody[movableLimbs.Length];
        limb_velocities = new float[movableLimbs.Length];

        for (int i = 0; i < movableLimbs.Length; i++)
        {
            rigidBodies[i] = ((GameObject) movableLimbs[i]).GetComponent(typeof(Rigidbody)) as Rigidbody;
        }

        thread = new Thread(this.ConnectionThread);
        thread.Start();
    }

    private void ConnectionThread()
    {
        // Model related logic
        connection = new ClientConnectionHandler();
        process = new Process();

        // Calls python training script.
        process.StartInfo.FileName = @"C:\Users\Michael\AppData\Local\Microsoft\WindowsApps\python.exe";
        string scriptPath = @"C:\Git\Virtual-Hand\PythonScripts\HandController.py";
        string modelName = "Real"; // TODO, remove hard coded at some point
        process.StartInfo.Arguments = scriptPath;

        // Starts the process
        print("Starting the process: " + process.StartInfo.FileName);
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.Start();
        System.Threading.Thread.Sleep(6000);

        connection.println(modelName);

        print("Started the Python process. ");
        print("From Python: '" + connection.readline() + "'");
        print("From Python: '" + connection.readline() + "'");

        while (running)
        {
            connection.println("next");
            limb_velocities = GeneralData.string2floatArray(connection.readline());
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Continuously applies existing torques to the limbs
        lock (limb_velocities_locker)
        {
            for (int i = 0; i < movableLimbs.Length; i++)
            {
                if (i % 3 != 0
                ) // TODO, if this works, then make a function that searches for the nearest parent with a rigid body (if exists)
                {
                    rigidBodies[i].angularVelocity =
                        new Vector3(
                            movableLimbs[i].transform.parent.GetComponent<Rigidbody>().angularVelocity.x +
                            limb_velocities[i], 0, 0);
                }
                else
                {
                    rigidBodies[i].angularVelocity = new Vector3(limb_velocities[i], 0, 0);
                }
            }
        }
    }

    private void OnDestroy()
    {
        connection.println("quit");
        running = false; // Stops the thread loop
        process.Close();
        connection.stop();
        print("Stopped.");
    }
}