#include <dlib/matrix.h>
#include <dlib/dnn.h>

using namespace dlib;

using weight_type = matrix<float, 1, 0>;
using sample_type = matrix<float, 0, 1>;

struct linear_regression
{
    linear_regression(int nparams, const std::string& name) : bias(0), name(name)
    {
        weights = matrix_cast<float>(gaussian_randm(1, nparams));
        bias = 0;
    }
    weight_type weights;
    float bias;
    const std::string name;
};

auto operator<<(std::ostream& sout, const linear_regression& model) -> std::ostream&
{
    sout << "name: " << model.name << '\n';
    sout << "weights:\n" << model.weights;
    sout << "bias: " << model.bias << '\n';
    return sout;
}

auto predict(const linear_regression& model, const sample_type& x) -> float
{
    return model.weights * x + model.bias;
}

auto loss(const linear_regression& model, const sample_type& x, const float y) -> float
{
    const auto temp = predict(model, x) - y;
    return temp * temp;
}

auto gradient(const linear_regression& model, const sample_type& x, const float y)
    -> std::pair<weight_type, float>
{
    // Here we should be using a 2, but instead, we use half the MSE, so that 2s cancel out.
    // That makes this implementation equivalent to dlib's one.
    // const weight_type gw = 2 * trans(x) * (wx + model.bias - y);
    // const float gb = 2 * (wx + model.bias - y);
    const float wx = model.weights * x;
    const weight_type gw = trans(x) * (wx + model.bias - y);
    const float gb = wx + model.bias - y;
    return {gw, gb};
}

auto sgd_update(
    linear_regression& model,
    const std::pair<weight_type, float>& grads,
    const float lr = 0.001)
{
    model.weights -= lr * grads.first;
    model.bias -= lr * grads.second;
}

int main()
try
{
    const weight_type weights_gt = {1.0, 2.7, 0.3, 1.2};
    const float bias_gt = 0.4;
    std::cout << "ground truth\n";
    std::cout << "weights:\n" << weights_gt << "bias: " << bias_gt << '\n';

    matrix<float> X = matrix_cast<float>(gaussian_randm(weights_gt.size(), 10'000));
    matrix<float> Y = weights_gt * X + bias_gt;
    X += 0.001 * matrix_cast<float>(gaussian_randm(X.nr(), X.nc()));
    // std::cout << "X: " << X.nr() << 'x' << X.nc() << '\n';
    // std::cout << "Y: " << Y.nr() << 'x' << Y.nc() << '\n';

    linear_regression model(4, "example");
    std::cout << model << '\n';

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < X.nc(); ++i)
    {
        const auto grads = gradient(model, colm(X, i), colm(Y, i));
        sgd_update(model, grads);
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << '\n';
    float final_loss = 0.0f;
    for (int i = 0; i < X.nc(); ++i)
    {
        final_loss += loss(model, colm(X, i), colm(Y, i));
    }

    std::cout << model << '\n';

    /**************************************************************************************/
    /*********************************** Deep learning ************************************/
    /**************************************************************************************/

    std::cin.get();
    std::cout << "\nDeep learning\n\n";
    // Network type definition:
    // - loss: mean squared error
    // - layer: fully connected with one output
    // - input: matrix of floats
    using net_type = loss_mean_squared<fc<1, input<matrix<float>>>>;
    net_type net;

    // helper function to print the parameters of our network
    auto print_params = [&net]()
    {
        const auto& params = net.subnet().layer_details().get_layer_params();
        const alias_tensor a_weights(params.size() - net.subnet().layer_details().get_num_outputs());
        const alias_tensor a_bias(net.subnet().layer_details().get_num_outputs());
        const tensor& weights = a_weights(params);
        const tensor& bias = a_bias(params, a_weights.size());
        std:: cout << "weights: \n";
        for (const auto& w : weights)
        {
            std::cout << w << ' ';
        }
        std::cout << "\nbias: " << *bias.begin() << '\n';
    };

    // We need to convert out samples into the right format for our network.
    std::vector<matrix<float>> samples;
    std::vector<float> labels;
    for (int i = 0; i < X.nc(); ++i)
    {
        samples.push_back(std::move(colm(X, i)));
        labels.push_back(colm(Y, i));
    }

    // initialize the network by forwarding one sample, so that we can see its
    // initial parameters
    net(samples[0]);
    print_params();

    // trainer for the network
    auto trainer = dlib::dnn_trainer(net, sgd(0, 0));
    trainer.set_learning_rate(0.001);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        trainer.train_one_step({samples[i]}, {labels[i]});
    }
    print_params();

    // We could also train the network with all the samples at once but, in that case,
    // we would need to increase the learning rate. Like this.
    // trainer.set_learning_rate(0.1);
    // trainer.set_max_num_epochs(1);
    // trainer.train(samples, labels);
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
