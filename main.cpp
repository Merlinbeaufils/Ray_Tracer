#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <math.h>
#include <random>
#include <iostream>

static std::default_random_engine engine(10);
static std::uniform_real_distribution<double> uniform(0, 1);


double gamma = 2.2;
int PER_PIXEL = 1;


class Vector {
public:
    double data[3];
    explicit Vector(double x = 0, double y = 0, double z = 0) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
        data[2] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

class Ray {
public:
    explicit Ray (Vector origin_, Vector direction_):
            origin(origin_), direction(direction_) {};

    Vector at_t(double t){
        return this -> origin + t * this -> direction;
    }

    Vector origin;
    Vector direction;
};

class Intersection {
public:
    Vector Point, N, color;
    double time;
    int found;
    int id;

    explicit Intersection(Vector P, Vector N_, Vector color_, int found_, int id_):
        Point(P), N(N_), color(color_), found(found_), id(id_) {};

    Intersection() {
        Point = Vector();
        N = Vector();
        color = Vector();
        found = id = 0;
        time = 0;
    }
};


class Geometry {
public:
    bool mirror = false, refraction = false, inverse_norm = false;
    int id = 0;
    Vector color = Vector();

    virtual Intersection inter(Ray ray);
};
Intersection Geometry::inter(Ray ray) {return Intersection();}

class Sphere : public Geometry{
public:

    Sphere (double r, Vector p, Vector color_) {
        radius = r;
        position = p;
        color = color_;
    }
    void make_mirror(){
        mirror = true;
    }


    void make_transparent(double n1 = 1.4){
        refraction = true;
        medium = n1;
    };
    void inverse(){
        inverse_norm = true;
    }

    Intersection inter(Ray ray);
    double radius, medium = 1;
    Vector position;



};


Intersection Sphere::inter(Ray ray){
    Intersection I;
    I.id = id;
    I.color = color;



    Vector u = ray.direction;
    Vector O = ray.origin;
    Vector C = position;
    double R = radius;
    Vector dist = O - C;

    double gradient = pow(dot(u, dist), 2) - dist.norm2() + pow(R, 2);

    if (gradient < 0){
        // no intersection
        return I;
    } else {
        double t1 = dot(u, -1 * dist) - sqrt(gradient);
        double t2 = t1 + 2 * sqrt(gradient);

        if (t2 < 0) {
            return I;
        } else if (t1 > 0) {
            I.time = t1;
            I.Point = ray.at_t(t1);
            I.found = true;
            I.N = I.Point - this -> position;
            I.N.normalize();
            return I;
        } else {
            I.time = t2;
            I.Point = ray.at_t(t2);
            I.found = true;
            I.N = I.Point - this -> position;
            I.N.normalize();
            return I;
        }
    }
}

class LightSource {
public:
    explicit LightSource (Vector position_, double intensity_):
            position(position_), intensity(intensity_) {};

    double intensity;
    Vector position;

};

//int ray_sphere_inter(Ray ray, Sphere sphere, double& time){
//
//    Vector u = ray.direction;
//    Vector O = ray.origin;
//    Vector C = sphere.position;
//    double R = sphere.radius;
//    Vector dist = O - C;
//
//    double gradient = pow(dot(u, dist), 2) - dist.norm2() + pow(R, 2);
//
//    if (gradient < 0){
//        // no intersection
//        return 0;
//    } else {
//        double t1 = dot(u, -1 * dist) - sqrt(gradient);
//        double t2 = t1 + 2 * sqrt(gradient);
//
//        if (t2 < 0) {
//            return 0;
//        } else if (t1 > 0) {
//            time = t1;
//            return 1;
//        } else {
//            time = t2;
//            return 1;
//        }
//    }
//}

class Scene {
public:
    LightSource light;
    std::vector<Geometry*> geometries;

    explicit Scene(LightSource light_): light(light_), geometries() {};


    void push_back(Geometry *object) {
        object -> id = geometries.size();
        geometries.push_back(object);
    }

    Intersection closest_inter(Ray ray, double& time);
    int visibility(Vector P, Vector light_p);
    Vector getColor(const Ray &ray, int ray_depth);

    bool indirect = true;
};

Intersection Scene::closest_inter(Ray ray, double& time) {
//    if (ray.origin.data[0] != 0){
//        std::cout << "x:  " << ray.origin.data[0] << "y: " << ray.origin.data[1] << "z: " << ray.origin.data[2] << '\n';
//    }
    Intersection I;
    for (Geometry *object: geometries){
        Intersection res = object -> inter(ray);
        if (res.found and (res.time < time or time < 0)) {
            I = res;
            time = res.time;
        }
    }
    return I;
}

int Scene::visibility(Vector P, Vector light_p){
    Vector direction = light_p - P;
    double d = direction.norm();

    direction.normalize();
    Ray light_ray = Ray(P, direction);

    double inter_time = -1;
    Intersection res = closest_inter(light_ray, inter_time);
    if (res.found) {
        return res.time > d - 10;
    } else {
        return true;
    }
}


Vector lambertian_scatter(Vector P, Vector albedo, Vector N, Scene scene) {
    Vector light_pos = scene.light.position;
    double intensity = scene.light.intensity;
    Vector omega = light_pos - P;
    double d = omega.norm();
    omega.normalize();
    N.normalize();

    Vector L = albedo * (intensity * scene.visibility(P + 1e-10 * N, light_pos) * fmax(dot(N, omega), 0)) / (4 * pow(M_PI,2) * pow(d, 2));
    //std::cout << L << '\n';
    return L;
}

Vector random_cos(Vector N) {
    //get a random vector for indirect lighting
    double r1 = uniform(engine), r2 = uniform(engine);
    double x = cos(2 * M_PI * r1) * pow(1 - r2, 0.5), y = sin(2 * M_PI * r1) * pow(1 - r2, 0.5);
    double z = sqrt(r2);
    Vector T1;

    int i,j;
    for (j = 1; i < 3; i++) {
        if (abs(N.data[j]) <= abs(N.data[i])) {
            i = j;
        }
    }
    //force the smallest component to 0, swap the two other components, and negate one
    //we proceed by case
    if (i == 0) T1 = Vector(0., -N[2], N[1]);
    if (i == 1) T1 = Vector(-N[2], 0., N[0]);
    if (i == 2) T1 = Vector(-N[1], N[0], 0.);
    //normalize
    T1.normalize();
    Vector T2 = cross(N, T1);
    T2.normalize();
    Vector V = x * T1 + y * T2 + z * N;
    V.normalize();
    return V;
}

Vector Scene::getColor(const Ray& ray , int ray_depth) {
    if (ray_depth < 0) return Vector();

    double time = -1;
    Intersection I = closest_inter(ray, time);
    if (I.found) {
        if (geometries[I.id] -> mirror) {
            Ray reflected_ray = Ray(I.Point + 1e-10 * I.N, ray.direction - 2 * dot(ray.direction, I.N) * I.N);
            return getColor(reflected_ray,ray_depth - 1);
        } else if (geometries[I.id] -> refraction) {
            if (geometries[I.id] -> inverse_norm){ I.N = -1 * I.N;}
            double n1 = 1., n2 = 1.4;
            Vector prev_N = I.N;
            if (dot(ray.direction,I.N) > 0) {
                I.N = (-1.)*I.N;
                n1 = 1.4;
                n2 = 1.;
            }

            double k0 = pow(n1 - n2, 2.) / pow(n1 + n2, 2.);
            I.Point = I.Point - I.N*1e-10;
            double ang = dot(ray.direction,I.N);
            if (1. - pow(n1/n2, 2.) * (1 - pow(ang, 2.)) > 0) {
                Vector tangential = (n1/n2) * (ray.direction - ang*I.N);
                Vector normal = (-1.)* I.N * sqrt(1 - pow((n1/n2),2.) * (1 - pow(ang,2.)));

                Vector transmitted = tangential + normal;
                double prob = ((double) rand() / (RAND_MAX));
                if (prob < k0 + (1 - k0)*pow(1 - abs(dot(I.N,transmitted)),5.)) {
                    Ray reflected_ray = Ray(I.Point, ray.direction - (2*dot(ray.direction,prev_N)) * prev_N);
                    return getColor(reflected_ray, ray_depth - 1);
                } else {
                    Ray refracted_ray = Ray(I.Point, transmitted);
                    return getColor(refracted_ray, ray_depth - 1);
                }
            } else {
                Ray internal = Ray(I.Point, ray.direction - (2*dot(ray.direction,prev_N)) * prev_N);
                return getColor(internal, ray_depth - 1);
            }





//            double n1 = 1, n2 = 1.4;
//            double ang = dot(ray.direction, N);
//            double offset = -1e-10;
//
//
//            double k0 = pow(0.4, 2.)/pow(2.4, 2.);
//
//            if (ang > 0) {N = -1 * N; n2 = 1.0; n1 = 1.4; }
//            double n1dn2 = n1/n2;
//
//
//
//            Vector tangential = n1dn2 * (ray.direction - ang * N);
//            Vector nor = -1 * N * sqrt(1 - pow(n1dn2, 2) * (1 - pow(dot(ray.direction, N),2)));
//            Ray transmitted_ray = Ray(P - offset * N, tangential + nor);
//            return getColor(transmitted_ray, ray_depth - 1);
        } else {
            Vector L = lambertian_scatter(I.Point, I.color, I.N, *this);

            if (!indirect) return L;

            Ray incoming = Ray(I.Point + 1e-10 * I.N, random_cos(I.N));
            I.color.normalize();

            Vector temp = getColor(incoming, ray_depth - 1);
            return L + Vector(temp[0] * I.color[0], temp[1] * I.color[1], temp[2] * I.color[2]);
        }
    }
};
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Vector get_normal(Vector P, Vector C){
    Vector distance = P - C;
    distance.normalize();
    return distance;
}





double correct(double x){
    x = fmin(255, x);
    x = x/255;
    x = pow(x, 1.0 / 2.2);
    x *= 255;
    return x;
}





int main() {
    // We use 512x512 pixels with angle 60 starting at origin (0,0,55);
    int W = 512;
    int H = 512;
    Vector Q = Vector(0,0,55);
    int alpha = 60 * M_PI / 180;
    double d = double(W) / (2 * tan(double(alpha) / 2));

    Sphere left_sphere = Sphere(9, Vector(-20, 0, 0), Vector(.2, .2, .2));
    left_sphere.make_mirror();
    Sphere center_sphere = Sphere(9, Vector(0, 0, 0), Vector(.2, .2, .2));
    center_sphere.make_transparent();
    Sphere right_sphere = Sphere(9, Vector(20, 0, 0), Vector(.2, .2, .2));
    right_sphere.make_transparent();
    Sphere right_sphere2 = Sphere(8.999, Vector(20, 0, 0), Vector(.2, .2, .2));
    right_sphere2.make_transparent();
    right_sphere2.inverse();
    Sphere green_sphere = Sphere(940, Vector(0,0,-1000), Vector(.1, .4, .1));
    Sphere red_sphere = Sphere(940, Vector(0,1000,0), Vector(.4, .1, .1));
    Sphere pink_sphere = Sphere(940, Vector(0,0,1000), Vector(.4, .1, .4));
    Sphere blue_sphere = Sphere(990, Vector(0,-1000,0), Vector(.1, .1, .4));
    Sphere yellow_sphere = Sphere(940, Vector(1000, 0, 0), Vector(.4,.4,.1));
    Sphere light_blue_sphere = Sphere(940, Vector(-1000, 0, 0), Vector(.4,.4,.9));

    LightSource light_source = LightSource(Vector(-10, 20, 40), 3E7);
    Scene scene = Scene(light_source);


    scene.push_back(&left_sphere);
    scene.push_back(&center_sphere);
    scene.push_back(&right_sphere);
    scene.push_back(&right_sphere2);

    scene.push_back(&green_sphere);
    scene.push_back(&red_sphere);
    scene.push_back(&pink_sphere);
    scene.push_back(&blue_sphere);
    scene.push_back(&light_blue_sphere);
    scene.push_back(&yellow_sphere);

    // double max_color = 0;

    std::vector<unsigned char> image(W * H * 3, 0);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {


            // build ray for given pixel
            double x_val = Q.data[0] + double(j) - double(W) / 2 + 0.5;
            double y_val = Q.data[1] + (double(H) - double(i) - 1) - double(H) / 2 + 0.5;
            double z_val = Q.data[2] - d;
            Vector direction = Vector(x_val, y_val, z_val);
            direction.normalize();
            Ray ray = Ray(Q, direction);

            Vector L;
            for (int i = 0; i < PER_PIXEL; i++){
                L = L + scene.getColor(ray,     5);
            }
            L = L / PER_PIXEL;

            image[(i * W + j) * 3 + 0] = correct(L.data[0]);
            image[(i * W + j) * 3 + 1] = correct(L.data[1]);
            image[(i * W + j) * 3 + 2] = correct(L.data[2]);

//            Vector P, N;
//            if (scene.closest_inter(ray, ray_inter_time, intersected, P, N)) {


//                Vector L = lambertian_scatter(P, intersected.color, N, scene);
//                //std::cout << L.data[1]<< '\n';
//                image[(i * W + j) * 3 + 0] = correct(L.data[0]);
//                image[(i * W + j) * 3 + 1] = correct(L.data[1]);
//                image[(i * W + j) * 3 + 2] = correct(L.data[2]);

//                double constant1 = fmax(dot(normal, omega), 0);
//
//                Vector slight_modif = (intersected.color / M_PI) * fmax(dot(normal, omega), 0);
//                slight_modif = slight_modif *
//                image[(i * W + j) * 3 + 0] = intersected.color.data[0] * 255;
//                image[(i * W + j) * 3 + 1] = intersected.color.data[1] * 255;
////                image[(i * W + j) * 3 + 2] = intersected.color.data[2] * 255;
//                image[(i * W + j) * 3 + 0] = slight_modif.data[0] * 255;
//                image[(i * W + j) * 3 + 1] = slight_modif.data[1] * 255;
//                image[(i * W + j) * 3 + 2] = slight_modif.data[2] * 255;

            // max_color = fmax(fmax(fmax(max_color, L.data[0]), L.data[1]), L.data[2]);
//            } else {
//                image[(i * W + j) * 3 + 0] = 0;
//                image[(i * W + j) * 3 + 1] = 0;
//                image[(i * W + j) * 3 + 2] = 0;
//            }
        }
    }
    stbi_write_png("image_indirect.png", W, H, 3, &image[0], 0);

    return 0;
}

